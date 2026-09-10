from qgis.core import (
    QgsApplication, QgsProject, QgsPrintLayout, QgsLayoutItemMap,
    QgsLayoutItemScaleBar, QgsLayoutSize, QgsLayoutPoint, QgsUnitTypes,
    QgsCoordinateReferenceSystem, QgsRectangle
)

QgsApplication.setPrefixPath("D:/QGIS/apps/qgis", True)
app = QgsApplication([], False)
app.initQgis()

project = QgsProject.instance()
project.read("d:/testrun/Mapathon/qgis/chennai_mapathon_ridership_desktop.qgs")

layout = project.layoutManager().layoutByName("19:8 Mapathon Final")
m = [i for i in layout.items() if isinstance(i, QgsLayoutItemMap)][0]
print("Map pos:", m.pos().x(), m.pos().y())
print("Map size:", m.rect().width(), m.rect().height())
print("Map item size:", m.sizeWithUnits().width(), m.sizeWithUnits().height())
print("Map extent:", m.extent().toString())
print("Map scale:", m.scale())
print("Map crs:", m.crs().authid())

# Now test creating a fresh layout properly
l2 = QgsPrintLayout(project)
l2.initializeDefaults()
page = l2.pageCollection().pages()[0]
page.setPageSize(QgsLayoutSize(430, 1030, QgsUnitTypes.LayoutMillimeters))

m2 = QgsLayoutItemMap(l2)
l2.addLayoutItem(m2)
m2.attemptMove(QgsLayoutPoint(15, 34, QgsUnitTypes.LayoutMillimeters))
m2.attemptResize(QgsLayoutSize(400, 950, QgsUnitTypes.LayoutMillimeters))
m2.setCrs(QgsCoordinateReferenceSystem("EPSG:32644"))

r = QgsRectangle(398500.0, 1403812.5, 427500.0, 1472687.5)
m2.setExtent(r)

print("\n--- Test m2 fresh ---")
print("m2 pos:", m2.pos().x(), m2.pos().y())
print("m2 rect:", m2.rect().width(), m2.rect().height())
print("m2 sizeWithUnits:", m2.sizeWithUnits().width(), m2.sizeWithUnits().height())
print("m2 extent:", m2.extent().toString())
print("m2 scale:", m2.scale())

sb2 = QgsLayoutItemScaleBar(l2)
sb2.setStyle("Single Box")
l2.addLayoutItem(sb2)
sb2.setLinkedMap(m2)
sb2.setUnits(QgsUnitTypes.DistanceKilometers)
sb2.setNumberOfSegments(4)
sb2.setUnitsPerSegment(2)
sb2.update()
print("sb2 mapUnitsPerScaleBarUnit:", sb2.mapUnitsPerScaleBarUnit())
print("sb2 unitLabel:", sb2.unitLabel())
print("sb2 rect:", sb2.rect().width(), sb2.rect().height())

from qgis.core import QgsLayoutExporter

exporter = QgsLayoutExporter(l2)
settings = QgsLayoutExporter.ImageExportSettings()
settings.dpi = 150
out_test = "d:/testrun/Mapathon/outputs/maps/test_scale_render.png"
result = exporter.exportToImage(out_test, settings)
print("Export result (0 is success):", result)
print("Scalebar rect:", sb2.rect().width(), sb2.rect().height())
print("Scalebar text:", sb2.displayName())

app.exitQgis()
