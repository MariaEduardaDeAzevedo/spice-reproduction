from typing import Dict, List, Set, Tuple
from spice.data_structure.scene_graph import SceneGraph
from spice.parser.parser import SpiceParser


class SPICE:
    def __init__(self, lang="en", wn_model=""):
        self.parser = SpiceParser(lang=lang, wn_model=wn_model)

    def compute_score(self, candidate: str, references: List[str]) -> Dict:
        """Compute SPICE score between candidate and references"""
        # Parse inputs
        cg = self.parser.parse(candidate).expand_synonyms()
        ref_graphs = [self.parser.parse(ref).expand_synonyms() for ref in references]

        # Convert graphs to tuples
        candidate_tuples = self._graph_to_tuples(cg)
        reference_tuples = set().union(*[self._graph_to_tuples(g) for g in ref_graphs])

        # Calculate scores
        matches = candidate_tuples & reference_tuples
        p = len(matches) / len(candidate_tuples) if candidate_tuples else 0
        r = len(matches) / len(reference_tuples) if reference_tuples else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            "candidate_tuples": candidate_tuples,
            "reference_tuples": reference_tuples,
        }

    def _graph_to_tuples(self, graph: SceneGraph) -> Set[Tuple]:
        """Convert scene graph to comparable tuples"""
        tuples = set()

        # Object tuples
        for obj in graph.objects:
            tuples.add(("object", obj))

        # Attribute tuples
        for obj, attrs in graph.attributes.items():
            for attr in attrs:
                tuples.add(("attribute", obj, attr))

        # Relation tuples
        for (subj, pred), objs in graph.relations.items():
            for obj in objs:
                tuples.add(("relation", subj, pred, obj))

        return tuples
