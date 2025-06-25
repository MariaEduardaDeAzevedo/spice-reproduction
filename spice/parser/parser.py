import stanza
from typing import List, Dict
from collections import defaultdict

import wn

import numpy as np
import subprocess
import logging

from spice.data_structure.scene_graph import SceneGraph


class SpiceParser:
    def __init__(self, lang="en", wn_model="oewn:2024"):
        stanza.download(lang)
        self.lang = lang
        self.wn_model = wn_model
        self.nlp = stanza.Pipeline(lang, processors="tokenize,pos,lemma,depparse")
        self.wordnet = None
        self.synonyms = defaultdict(set)

        try:
            self.wordnet = wn.Wordnet(self.wn_model)
        except Exception as e:
            try:
                wn.download(self.wn_model)
                self.wordnet = wn.Wordnet(self.wn_model)
            except Exception as e:
                raise e

        self._load_synonyms()

    def _load_synonyms(self):
        for synset in self.wordnet.synsets():

            lemmas = synset.lemmas()

            for lemma in lemmas:
                lemma_lower = lemma.lower()

                self.synonyms[lemma_lower].update(
                    [l.lower() for l in lemmas if l != lemma]
                )

                self.synonyms[lemma_lower].add(lemma_lower)

            for related in synset.relations("equivalent"):
                related_lemmas = related.target().lemmas()
                for lemma in lemmas:
                    lemma_lower = lemma.lower()
                    self.synonyms[lemma_lower].update(
                        [rl.lower() for rl in related_lemmas]
                    )

    def parse(self, sentence: str) -> SceneGraph:
        doc = self.nlp(sentence)
        graph = SceneGraph(
            objects=set(),
            attributes=defaultdict(set),
            relations=defaultdict(set),
            synonyms=self.synonyms,
        )

        for sent in doc.sentences:
            idx_to_word = {word.id: word for word in sent.words}
            dependencies = [(word.head, word.deprel, word.id) for word in sent.words]

            self._extract_objects(
                graph,
                sent.words,
            )
            self._extract_attributes(graph, sent.words)
            self._extract_relations(graph, sent.words, idx_to_word)

        return graph

    def _extract_objects(self, graph: SceneGraph, words: List) -> None:
        for word in words:
            # Rule 1: Nouns/proper nouns not in modifier role
            if word.upos in ["NOUN", "PROPN"] and not word.deprel.endswith("mod"):
                graph.objects.add(word.lemma.lower())

            # Rule 2: Nominal clauses introducing new entities
            if word.deprel in ["acl:relcl", "acl", "ccomp", "xcomp"]:
                graph.objects.add(word.lemma.lower())

    def _extract_attributes(self, graph: SceneGraph, words: List) -> None:

        for idx, word in enumerate(words):
            head_idx = word.head
            head_word = words[head_idx - 1]
            if word.upos in ["ADJ"] or word.deprel in [
                "amod",
                "nummod",
                "advmod",
                "appos",
                "acl",
                "acl:relcl",
            ]:
                if head_idx == 0:
                    continue
                if head_word.upos in ["NOUN", "PROPN"]:
                    noun = head_word.lemma.lower()
                    adj = word.lemma.lower()
                    graph.attributes[noun].add(adj)
            elif word.deprel == "case" and word.lemma.lower() in [
                "de",
                "da",
                "do",
                "com",
                "em",
            ]:
                head = words[word.head - 1]
                if head.upos in ["NOUN", "PROPN"]:
                    children = [w for w in words if word.id - 1 in w.parent.id]
                    for c in children:
                        graph.attributes[c.lemma.lower()].add(head.lemma.lower())

    def _extract_relations(
        self, graph: SceneGraph, words: List, idx_to_word: Dict
    ) -> None:
        for word in words:
            if word.upos != "VERB":
                continue

            subj, obj = None, None
            predicate = word.lemma.lower()

            children = [w for w in words if w.head == word.id]

            # Process verb arguments
            for child in children:
                # Core arguments
                if child.deprel == "nsubj":
                    subj = child.lemma.lower()
                elif child.deprel == "obj":
                    obj = child.lemma.lower()

                # Adverbial clauses
                elif child.deprel == "advcl":
                    graph.relations[(predicate, "advcl")].add(child.lemma.lower())

                # Oblique arguments with prepositions
                elif child.deprel.startswith("obl"):
                    c_children = [w for w in words if w.head == child.id]
                    prep = next(
                        (c.lemma.lower() for c in c_children if c.deprel == "case"),
                        None,
                    )
                    if prep:
                        graph.relations[(subj or predicate, prep)].add(
                            child.lemma.lower()
                        )

                # Copula handling
                elif child.deprel == "cop" and subj:
                    graph.relations[(subj, "ser")].add(predicate)

                # Relative clauses
                elif child.deprel == "acl:relcl" and subj:
                    graph.relations[(subj, "que")].add(predicate)

            # Core proposition
            if subj and obj:
                graph.relations[(subj, predicate)].add(obj)

    def _handle_special_constructions(
        self, graph: SceneGraph, dependencies: List, idx_to_word: Dict
    ) -> None:
        # for head_idx, rel, dep_idx in dependencies:
        #     head_word = idx_to_word[head_idx] if head_idx != 0 else None
        #     dep_word = idx_to_word[dep_idx]

        #     if rel in ["conj", "appos"]:

        #         head_lemma = head_word.lemma.lower() if head_word else ""
        #         dep_lemma = dep_word.lemma.lower()
        #         graph.objects.add(dep_lemma)

        #         if head_lemma in graph.attributes:
        #             graph.attributes[dep_lemma].update(graph.attributes[head_lemma])

        #         for (subj, pred), objs in list(graph.relationsitems()):
        #             if subj == head_lemma:
        #                 graph.relations[(dep_lemma, pred)].update(objs)

        #             elif rel == "aux":
        #                 graph.relations = {
        #                     (subj, f"{head_lemma}_{dep_word.lemma.lower()}"): (
        #                         objs
        #                         if subj
        #                         else (
        #                             head_lemma + "_" + dep_word.lemma.lower(),
        #                             objs,
        #                         )
        #                     )
        #                 }

        #     elif rel == "nmod:poss":

        #         possessor = dep_word.lemma.lower()
        #         possessed = head_word.lemma.lower() if head_word else ""
        #         if possessed:
        #             graph.relations[(possessor, "possess")].add(possessed)

        """Handle conjunctions, appositions, auxiliaries, and possession"""
        for head_idx, rel, dep_idx in dependencies:
            head_word = idx_to_word.get(head_idx)
            dep_word = idx_to_word.get(dep_idx)

            if not head_word or not dep_word:
                continue

            head_lemma = head_word.lemma.lower()
            dep_lemma = dep_word.lemma.lower()

            # Conjunctions and appositions
            if rel in ["conj", "appos"]:
                # Propagate attributes
                if head_lemma in graph.attributes:
                    graph.attributes[dep_lemma] |= graph.attributes[head_lemma]

                # Propagate relations
                for (subj, pred), objs in list(graph.relations.items()):
                    if subj == head_lemma:
                        graph.relations[(dep_lemma, pred)] |= objs

            # Auxiliary verbs
            elif rel == "aux":
                new_predicate = f"{head_lemma}_{dep_lemma}"
                # Update existing relations
                for (subj, pred), objs in list(graph.relations.items()):
                    if pred == head_lemma:
                        graph.relations[(subj, new_predicate)] = objs
                        del graph.relations[(subj, pred)]

            # Possessive constructions
            elif rel == "nmod:poss":
                graph.relations[(dep_lemma, "possess")].add(head_lemma)
