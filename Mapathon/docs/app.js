/**
 * Chennai Metro Phase 1 & 2 Interactive Web Map
 * ISRO Bhuvan Mapathon 2026
 * Lead: Subhash B | Mentor: Ruthvika R
 */

(function () {
  'use strict';

  // --- Configuration & Constants ---
  const CHENNAI_CENTER = [13.040, 80.215];
  const DEFAULT_ZOOM = 11;
  const CHENNAI_BOUNDS = [
    [12.80, 79.98],
    [13.26, 80.35]
  ];

  const LINE_COLORS = {
    'Blue Line': '#0066cc',
    'Green Line': '#009944',
    'Purple Line': '#6f42c1',
    'Yellow Line': '#fd7e14',
    'Red Line': '#dc3545'
  };

  const LULC_COLORS = {
    'Built-up': { fill: '#e6550d', stroke: '#a63603' },
    'Water/Wetland': { fill: '#2b83ba', stroke: '#1d5a80' },
    'Agriculture': { fill: '#fed976', stroke: '#feb24c' },
    'Forest/Scrub': { fill: '#2ca25f', stroke: '#006d2c' },
    'Barren/Open': { fill: '#dfc27d', stroke: '#bf812d' }
  };

  // --- State ---
  let map;
  let baseLayers = {};
  let currentBaseLayer = 'positron';
  let layers = {
    lulc: null,
    catchments: null,
    corridors: null,
    stations: null
  };
  let rawData = {
    corridors: null,
    stations: null,
    catchments: null,
    lulc: null
  };
  let activeFilters = {
    phase: 'all',
    corridor: 'all',
    top15Only: false
  };
  let stationMarkers = [];
  let currentActiveMarker = null;

  // --- Initialize Application ---
  window.addEventListener('DOMContentLoaded', async () => {
    initMap();
    initBasemaps();
    setupEventListeners();
    await loadData();
    renderLayers();
    populateLeaderboard();
  });

  // --- Map Initialization ---
  function initMap() {
    map = L.map('map', {
      center: CHENNAI_CENTER,
      zoom: DEFAULT_ZOOM,
      minZoom: 9,
      maxZoom: 18,
      zoomControl: false
    });

    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.control.scale({ imperial: false, position: 'bottomleft' }).addTo(map);
  }

  // --- Basemap Setup ---
  function initBasemaps() {
    baseLayers.positron = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>, &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      subdomains: 'abcd',
      maxZoom: 19
    }).addTo(map);

    baseLayers.darkmatter = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>, &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      subdomains: 'abcd',
      maxZoom: 19
    });

    baseLayers.satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri, Maxar, Earthstar Geographics',
      maxZoom: 18
    });

    baseLayers.osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19
    });

    // Handle Basemap Switcher Buttons
    const buttons = document.querySelectorAll('.btn-basemap');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.dataset.basemap;
        if (target === currentBaseLayer) return;

        buttons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        map.removeLayer(baseLayers[currentBaseLayer]);
        map.addLayer(baseLayers[target]);
        currentBaseLayer = target;
      });
    });
  }

  // --- Data Loading (JS Bundle or GeoJSON fetch) ---
  async function loadData() {
    if (window.METRO_DATA) {
      rawData = window.METRO_DATA;
      return;
    }

    try {
      const [corridors, stations, catchments, lulc] = await Promise.all([
        fetch('data/corridors.geojson').then(r => r.json()),
        fetch('data/stations.geojson').then(r => r.json()),
        fetch('data/catchments.geojson').then(r => r.json()),
        fetch('data/lulc.geojson').then(r => r.json())
      ]);
      rawData = { corridors, stations, catchments, lulc };
    } catch (err) {
      console.error('Error loading GeoJSON data:', err);
    }
  }

  // --- Rendering Layers ---
  function renderLayers() {
    renderLulcLayer();
    renderCatchmentsLayer();
    renderCorridorsLayer();
    renderStationsLayer();
  }

  // 1. ISRO Bhuvan LULC
  function renderLulcLayer() {
    if (!rawData.lulc) return;

    const opacity = document.getElementById('lulcOpacity').value / 100;

    layers.lulc = L.geoJSON(rawData.lulc, {
      style: (feature) => {
        const group = feature.properties.lulc_group || 'Barren/Open';
        const color = LULC_COLORS[group] || LULC_COLORS['Barren/Open'];
        return {
          fillColor: color.fill,
          color: color.stroke,
          weight: 0.6,
          fillOpacity: opacity * 0.75,
          opacity: opacity
        };
      },
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        layer.bindTooltip(`
          <strong>ISRO Bhuvan LULC:</strong> ${p.lulc_group || 'N/A'}<br>
          <small>${p.lulc_class || ''} (${p.map_year || '2015-16'})</small>
        `, { sticky: true, opacity: 0.9 });
      }
    });

    if (document.getElementById('layerToggleLulc').checked) {
      layers.lulc.addTo(map);
    }
  }

  // 2. 800m Walking Catchments
  function renderCatchmentsLayer() {
    if (!rawData.catchments) return;

    const opacity = document.getElementById('catchmentOpacity').value / 100;

    layers.catchments = L.geoJSON(rawData.catchments, {
      filter: (feature) => filterStationFeature(feature.properties),
      style: (feature) => {
        const line = feature.properties.Line;
        const color = LINE_COLORS[line] || '#0284c7';
        return {
          fillColor: color,
          color: color,
          weight: 1.2,
          dashArray: '3, 4',
          fillOpacity: opacity * 0.45,
          opacity: opacity * 0.8
        };
      },
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        layer.bindTooltip(`<strong>${p.name}</strong><br><small>800m Pedestrian Catchment</small>`, {
          sticky: true,
          opacity: 0.85
        });
      }
    });

    if (document.getElementById('layerToggleCatchments').checked) {
      layers.catchments.addTo(map);
    }
  }

  // 3. Metro Corridors (Alignment Lines)
  function renderCorridorsLayer() {
    if (!rawData.corridors) return;

    layers.corridors = L.geoJSON(rawData.corridors, {
      filter: (feature) => {
        if (activeFilters.corridor !== 'all') {
          return feature.properties.Line === activeFilters.corridor;
        }
        if (activeFilters.phase !== 'all') {
          return feature.properties.phase === activeFilters.phase;
        }
        return true;
      },
      style: (feature) => {
        const line = feature.properties.Line;
        const color = LINE_COLORS[line] || '#0284c7';
        const isPhase1 = feature.properties.phase === 'Phase 1';
        return {
          color: color,
          weight: isPhase1 ? 5 : 4,
          opacity: 0.85,
          dashArray: isPhase1 ? null : '6, 4'
        };
      },
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        layer.bindTooltip(`
          <strong style="color:${LINE_COLORS[p.Line] || '#000'}">${p.Line}</strong><br>
          <small>${p.phase} &bull; ${p.length_km ? p.length_km.toFixed(1) + ' km' : ''}</small>
        `, { sticky: true });
      }
    });

    if (document.getElementById('layerToggleLines').checked) {
      layers.corridors.addTo(map);
    }
  }

  // 4. Metro Stations (Graduated Bubbles & Hubs)
  function renderStationsLayer() {
    if (!rawData.stations) return;

    if (layers.stations) {
      map.removeLayer(layers.stations);
    }
    stationMarkers = [];

    layers.stations = L.geoJSON(rawData.stations, {
      filter: (feature) => filterStationFeature(feature.properties),
      pointToLayer: (feature, latlng) => {
        const p = feature.properties;
        const isHub = !!p.is_intermodal_hub;
        const isTop1 = !!p.is_top1_phase2;
        const isPhase1 = p.phase === 'Phase 1';
        const prediction = p.prediction || 0;

        // Radius calculation based on boardings
        let radius = 6;
        if (prediction > 15000) radius = 13;
        else if (prediction > 10000) radius = 11;
        else if (prediction > 7500) radius = 9.5;
        else if (prediction > 5000) radius = 7.5;
        else radius = 5.5;

        // Color based on Line and prediction tier
        const lineColor = LINE_COLORS[p.Line] || '#0284c7';
        let fillColor = '#ffffcc';
        if (prediction > 12000) fillColor = '#225ea8';
        else if (prediction > 7500) fillColor = '#41b6c4';
        else if (prediction > 5000) fillColor = '#7fcdbb';
        else fillColor = '#c7e9b4';

        if (isPhase1) {
          fillColor = lineColor;
        }

        const markerOptions = {
          radius: isHub ? radius + 2 : radius,
          fillColor: fillColor,
          color: isHub ? '#f1c40f' : (isPhase1 ? '#ffffff' : lineColor),
          weight: isHub ? 3 : (isTop1 ? 3 : 1.8),
          opacity: 1,
          fillOpacity: isPhase1 ? 0.9 : 0.85
        };

        const marker = L.circleMarker(latlng, markerOptions);
        marker.featureData = p;
        marker.latlng = latlng;
        stationMarkers.push(marker);

        // Bind interactive tooltip & click
        marker.bindTooltip(`
          <strong>${p.name}</strong><br>
          <span style="color:${lineColor}">● ${p.Line}</span> &bull; ${p.phase}<br>
          <b>${prediction ? prediction.toLocaleString() : 'N/A'}</b> boardings/day
          ${p.rank_phase2 ? `<br><small style="color:#e6550d;font-weight:700">#${Math.round(p.rank_phase2)} in Phase 2</small>` : ''}
        `, { direction: 'top', offset: [0, -6] });

        marker.on('click', () => {
          selectStation(marker);
        });

        return marker;
      }
    });

    if (document.getElementById('layerToggleStations').checked) {
      layers.stations.addTo(map);
    }
  }

  // Helper: Filter Station Feature
  function filterStationFeature(p) {
    // Phase filter
    if (activeFilters.phase !== 'all' && p.phase !== activeFilters.phase) {
      return false;
    }
    // Corridor filter
    if (activeFilters.corridor !== 'all' && p.Line !== activeFilters.corridor) {
      return false;
    }
    // Top 15 filter
    if (activeFilters.top15Only) {
      return p.rank_phase2 && p.rank_phase2 <= 15;
    }
    return true;
  }

  // --- Station Selection & Details Drawer ---
  function selectStation(marker) {
    const p = marker.featureData;
    currentActiveMarker = marker;

    // Center map slightly offset for the bottom drawer
    map.flyTo(marker.latlng, Math.max(map.getZoom(), 14), { duration: 0.8 });

    // Open detail drawer
    const drawer = document.getElementById('stationDetailDrawer');
    const badge = document.getElementById('drawerLineBadge');
    const nameEl = document.getElementById('drawerStationName');
    const bodyEl = document.getElementById('drawerBody');

    const lineColor = LINE_COLORS[p.Line] || '#0284c7';
    badge.textContent = `${p.Line} (${p.phase})`;
    badge.style.backgroundColor = lineColor;
    nameEl.textContent = p.name;

    const isPhase1 = p.phase === 'Phase 1';
    const valLabel = isPhase1 ? 'Observed Daily Boardings (CMRL Actual)' : 'Forecasted Daily Boardings (Model)';
    const predictionVal = p.prediction ? Math.round(p.prediction).toLocaleString() : 'N/A';
    const ciText = (p.lower && p.upper) ? `90% CI: ${Math.round(p.lower).toLocaleString()} – ${Math.round(p.upper).toLocaleString()}` : '';

    bodyEl.innerHTML = `
      <div class="detail-forecast-box">
        <div class="detail-forecast-value">${predictionVal}</div>
        <div class="detail-forecast-label">${valLabel}</div>
        ${ciText ? `<div class="detail-ci">${ciText}</div>` : ''}
        ${p.rank_phase2 ? `<div style="margin-top:6px;font-weight:700;color:#d97706;"><i class="fa-solid fa-trophy"></i> Rank #${Math.round(p.rank_phase2)} Demand in Phase 2</div>` : ''}
        ${p.is_intermodal_hub ? `<div style="margin-top:4px;font-weight:700;color:#059669;"><i class="fa-solid fa-star"></i> Multimodal Interchange Hub</div>` : ''}
      </div>

      <div class="detail-metrics-grid">
        <div class="metric-chip">
          <div class="metric-chip-title"><i class="fa-solid fa-bus"></i> Feeder Bus Stops (500m)</div>
          <div class="metric-chip-value">${p.bus_stops_500m !== undefined ? p.bus_stops_500m : 'N/A'}</div>
        </div>
        <div class="metric-chip">
          <div class="metric-chip-title"><i class="fa-solid fa-users"></i> Pop. Density (GHSL)</div>
          <div class="metric-chip-value">${p.population_density ? Math.round(p.population_density).toLocaleString() + '/km²' : 'N/A'}</div>
        </div>
        <div class="metric-chip">
          <div class="metric-chip-title"><i class="fa-solid fa-location-dot"></i> Surrounding POIs</div>
          <div class="metric-chip-value">${p.poi_total !== undefined ? p.poi_total : 'N/A'}</div>
        </div>
        <div class="metric-chip">
          <div class="metric-chip-title"><i class="fa-solid fa-square-parking"></i> Park & Ride</div>
          <div class="metric-chip-value">${p.pnr ? '<span style="color:#16a34a;">Available</span>' : '<span style="color:#94a3b8;">None</span>'}</div>
        </div>
      </div>

      <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4; border-top: 1px solid #f1f5f9; padding-top: 8px;">
        <strong>Catchment Analysis:</strong> 800m Euclidean walking buffer overlaid with ISRO Bhuvan LULC 1:50,000 Level-II land use classes.
      </div>
    `;

    drawer.style.display = 'block';
  }

  // --- Leaderboard Population ---
  function populateLeaderboard() {
    if (!rawData.stations) return;

    const list = document.getElementById('leaderboardList');
    list.innerHTML = '';

    // Filter phase 2 stations that have ranks
    const phase2Stations = rawData.stations.features
      .filter(f => f.properties.phase === 'Phase 2' && f.properties.rank_phase2)
      .sort((a, b) => a.properties.rank_phase2 - b.properties.rank_phase2)
      .slice(0, 10);

    phase2Stations.forEach(f => {
      const p = f.properties;
      const rank = Math.round(p.rank_phase2);
      const isTop3 = rank <= 3;
      const lineColor = LINE_COLORS[p.Line] || '#0284c7';

      const item = document.createElement('div');
      item.className = 'leaderboard-item';
      item.innerHTML = `
        <div class="leaderboard-left">
          <div class="rank-badge ${isTop3 ? 'top3' : ''}">${rank}</div>
          <div>
            <div class="leaderboard-name">${p.name}</div>
            <div class="leaderboard-sub" style="color:${lineColor}">● ${p.Line}</div>
          </div>
        </div>
        <div class="leaderboard-value">${p.prediction ? Math.round(p.prediction).toLocaleString() : ''}</div>
      `;

      item.addEventListener('click', () => {
        const marker = stationMarkers.find(m => m.featureData.name === p.name);
        if (marker) {
          selectStation(marker);
        }
      });

      list.appendChild(item);
    });
  }

  // --- Event Listeners Setup ---
  function setupEventListeners() {
    // 1. Sidebar Toggle
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebarToggleBtn');
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      setTimeout(() => map.invalidateSize(), 310);
    });

    // 2. Reset View
    document.getElementById('resetViewBtn').addEventListener('click', () => {
      map.fitBounds(CHENNAI_BOUNDS);
    });

    // 3. Close Station Drawer
    document.getElementById('closeDrawerBtn').addEventListener('click', () => {
      document.getElementById('stationDetailDrawer').style.display = 'none';
    });

    // 4. Team Modal
    const teamModal = document.getElementById('teamModal');
    document.getElementById('teamInfoBtn').addEventListener('click', () => {
      teamModal.style.display = 'flex';
    });
    document.getElementById('closeModalBtn').addEventListener('click', () => {
      teamModal.style.display = 'none';
    });
    document.getElementById('closeModalFooterBtn').addEventListener('click', () => {
      teamModal.style.display = 'none';
    });
    teamModal.addEventListener('click', (e) => {
      if (e.target === teamModal) teamModal.style.display = 'none';
    });

    // 5. Search Bar & Dropdown
    const searchInput = document.getElementById('stationSearchInput');
    const searchDropdown = document.getElementById('searchResultsDropdown');
    const clearBtn = document.getElementById('clearSearchBtn');

    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.trim().toLowerCase();
      if (!q) {
        searchDropdown.style.display = 'none';
        clearBtn.style.display = 'none';
        return;
      }

      clearBtn.style.display = 'block';

      if (!rawData.stations) return;

      const matches = rawData.stations.features
        .filter(f => f.properties.name && f.properties.name.toLowerCase().includes(q))
        .slice(0, 8);

      if (matches.length === 0) {
        searchDropdown.innerHTML = '<div style="padding:10px;font-size:0.8rem;color:#94a3b8;">No matching stations</div>';
        searchDropdown.style.display = 'block';
        return;
      }

      searchDropdown.innerHTML = '';
      matches.forEach(f => {
        const p = f.properties;
        const item = document.createElement('div');
        item.className = 'search-result-item';
        item.innerHTML = `
          <div>
            <div class="search-item-name">${p.name}</div>
            <div class="search-item-sub">${p.Line} &bull; ${p.phase}</div>
          </div>
          <span class="search-item-badge" style="background:${LINE_COLORS[p.Line]}15;color:${LINE_COLORS[p.Line]};">
            ${p.prediction ? Math.round(p.prediction).toLocaleString() : ''}
          </span>
        `;

        item.addEventListener('click', () => {
          searchDropdown.style.display = 'none';
          searchInput.value = p.name;
          const marker = stationMarkers.find(m => m.featureData.name === p.name);
          if (marker) {
            selectStation(marker);
          }
        });

        searchDropdown.appendChild(item);
      });

      searchDropdown.style.display = 'block';
    });

    clearBtn.addEventListener('click', () => {
      searchInput.value = '';
      searchDropdown.style.display = 'none';
      clearBtn.style.display = 'none';
    });

    // 6. Phase Filters
    const phaseBtns = document.querySelectorAll('#phaseFilterGroup .btn-pill');
    phaseBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        phaseBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFilters.phase = btn.dataset.phase;
        applyFilters();
      });
    });

    // 7. Corridor Filter Select
    document.getElementById('corridorSelect').addEventListener('change', (e) => {
      activeFilters.corridor = e.target.value;
      applyFilters();
    });

    // 8. Top 15 Toggle
    document.getElementById('top15Toggle').addEventListener('change', (e) => {
      activeFilters.top15Only = e.target.checked;
      applyFilters();
    });

    // 9. Layer Visibility Toggles
    document.getElementById('layerToggleStations').addEventListener('change', (e) => {
      if (e.target.checked) map.addLayer(layers.stations);
      else map.removeLayer(layers.stations);
    });

    document.getElementById('layerToggleLines').addEventListener('change', (e) => {
      if (e.target.checked) map.addLayer(layers.corridors);
      else map.removeLayer(layers.corridors);
    });

    document.getElementById('layerToggleCatchments').addEventListener('change', (e) => {
      if (e.target.checked) map.addLayer(layers.catchments);
      else map.removeLayer(layers.catchments);
    });

    document.getElementById('layerToggleLulc').addEventListener('change', (e) => {
      if (e.target.checked) map.addLayer(layers.lulc);
      else map.removeLayer(layers.lulc);
    });

    // 10. Opacity Sliders
    const catchmentSlider = document.getElementById('catchmentOpacity');
    const catchmentVal = document.getElementById('catchmentOpacityVal');
    catchmentSlider.addEventListener('input', (e) => {
      const val = e.target.value;
      catchmentVal.textContent = `${val}%`;
      if (layers.catchments) {
        layers.catchments.setStyle({
          fillOpacity: (val / 100) * 0.45,
          opacity: (val / 100) * 0.8
        });
      }
    });

    const lulcSlider = document.getElementById('lulcOpacity');
    const lulcVal = document.getElementById('lulcOpacityVal');
    lulcSlider.addEventListener('input', (e) => {
      const val = e.target.value;
      lulcVal.textContent = `${val}%`;
      if (layers.lulc) {
        layers.lulc.setStyle({
          fillOpacity: (val / 100) * 0.75,
          opacity: val / 100
        });
      }
    });
  }

  // --- Apply Filters Across All Layers ---
  function applyFilters() {
    renderCatchmentsLayer();
    renderCorridorsLayer();
    renderStationsLayer();

    // Update KPI card count
    let visibleCount = stationMarkers.length;
    document.getElementById('kpiTotalStations').textContent = visibleCount;
  }

})();
