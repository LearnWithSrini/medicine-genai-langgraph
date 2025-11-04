from rdflib import Graph
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def build_graph():
    g = Graph()
    g.parse(os.path.join(BASE_DIR, "graph", "bi_ontology.ttl"), format="turtle")
    g.parse(os.path.join(BASE_DIR, "graph", "bi_molecule_data.ttl"), format="turtle")
    out_path = os.path.join(BASE_DIR, "graph", "bi_all.nt")
    g.serialize(out_path, format="nt")
    print(f"Combined RDF written to {out_path}")
    print("Upload this file with your RDF database bulk loader.")

if __name__ == "__main__":
    build_graph()
