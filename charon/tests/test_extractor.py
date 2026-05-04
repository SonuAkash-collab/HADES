from __future__ import annotations

from shared import extractor as extractor_module
from shared.triple import KnowledgeTriple


class FakeToken:
    def __init__(
        self,
        text: str,
        dep_: str = "",
        pos_: str = "",
        lemma_: str = "",
        head=None,
        children=None,
        subtree=None,
    ) -> None:
        self.text = text
        self.dep_ = dep_
        self.pos_ = pos_
        self.lemma_ = lemma_ or text
        self.head = head
        self.children = children or []
        self.subtree = subtree if subtree is not None else [self]


class FakeSentence:
    def __init__(self, tokens):
        self._tokens = tokens

    def __iter__(self):
        return iter(self._tokens)


class FakeDoc:
    def __init__(self, sentences):
        self.sents = sentences


class FakeNLP:
    def __init__(self, doc):
        self._doc = doc

    def __call__(self, text: str):
        return self._doc


def test_extract_source_triples(monkeypatch):
    sent = FakeSentence(
        [
            FakeToken("Caesar", dep_="nsubj"),
            FakeToken("conquered", dep_="ROOT", pos_="VERB", lemma_="conquer"),
            FakeToken("Gaul", dep_="dobj"),
        ]
    )
    monkeypatch.setattr(extractor_module, "_load_spacy_model", lambda: FakeNLP(FakeDoc([sent])))

    triples = extractor_module.extract_source_triples("Caesar conquered Gaul.")

    assert triples == [KnowledgeTriple(subject="Caesar", verb="conquer", object="Gaul")]


def test_extract_object_with_verb_prep_phrase(monkeypatch):
    subj = FakeToken("It", dep_="nsubj")
    verb = FakeToken("generates", dep_="ROOT", pos_="VERB", lemma_="generate")
    dobj = FakeToken("ATP", dep_="dobj", head=verb, subtree=[FakeToken("ATP")])
    prep = FakeToken("through", dep_="prep", head=verb)
    pobj = FakeToken(
        "phosphorylation",
        dep_="pobj",
        head=prep,
        subtree=[FakeToken("through"), FakeToken("oxidative"), FakeToken("phosphorylation")],
    )

    prep.subtree = [FakeToken("through"), FakeToken("oxidative"), FakeToken("phosphorylation")]
    prep.children = [pobj]
    verb.children = [dobj, prep]

    sent = FakeSentence([subj, verb, dobj, prep, pobj])
    monkeypatch.setattr(extractor_module, "_load_spacy_model", lambda: FakeNLP(FakeDoc([sent])))

    triples = extractor_module.extract_source_triples("It generates ATP through oxidative phosphorylation.")

    assert triples == [KnowledgeTriple(subject="It", verb="generate", object="ATP through oxidative phosphorylation")]


def test_extract_object_keeps_compound_nouns(monkeypatch):
    sent = FakeSentence(
        [
            FakeToken("Armstrong", dep_="nsubj", subtree=[FakeToken("Neil"), FakeToken("Armstrong")]),
            FakeToken("piloted", dep_="ROOT", pos_="VERB", lemma_="pilot"),
            FakeToken(
                "Module",
                dep_="dobj",
                subtree=[FakeToken("the"), FakeToken("Lunar"), FakeToken("Module"), FakeToken("Eagle")],
            ),
        ]
    )
    monkeypatch.setattr(extractor_module, "_load_spacy_model", lambda: FakeNLP(FakeDoc([sent])))

    triples = extractor_module.extract_source_triples("Neil Armstrong piloted the Lunar Module Eagle.")

    assert triples == [KnowledgeTriple(subject="Neil Armstrong", verb="pilot", object="the Lunar Module Eagle")]
