import io
import pydotplus
from rdflib.tools.rdf2dot import rdf2dot
from PIL import Image

def visualize(g):
    stream = io.StringIO()
    rdf2dot(g, stream)
    dg = pydotplus.graph_from_dot_data(stream.getvalue())
    png = dg.create_png()
    with open("graph.png", "wb") as f:
        f.write(png)
    img = Image.open("graph.png")
    img.show()

