from collections import defaultdict
from typing import Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SceneGraph:
    objects: Set[str]
    attributes: Dict[str, Set[str]]
    relations: Dict[Tuple[str, str], Set[str]]
    synonyms: Dict[str, Set[str]]

    def expand_synonyms(self):
        expanded = SceneGraph(
            objects=set(),
            attributes=defaultdict(set),
            relations=defaultdict(set),
            synonyms=self.synonyms,
        )

        for obj in self.objects:
            expanded.objects.update(self.synonyms.get(obj, {obj}))

        for obj, attrs in self.attributes.items():
            for syn_obj in self.synonyms.get(obj, {obj}):
                expanded.attributes[syn_obj].update(
                    {syn for attr in attrs for syn in self.synonyms.get(attr, {attr})}
                )

        for (subj, pred), objs in self.relations.items():
            for syn_subj in self.synonyms.get(subj, {subj}):
                for syn_pred in self.synonyms.get(pred, {pred}):
                    expanded.relations[(syn_subj, syn_pred)].update(
                        {syn for obj in objs for syn in self.synonyms.get(obj, {obj})}
                    )

        return expanded
