from spacy.matcher import DependencyMatcher
from spacy.pipeline import merge_entities


def construct_pattern(dependency_triples: List[List[str]]):
    """
    Idea: add patterns to a matcher designed to find a subtree in a
    spacy dependency tree. Rules are strictly of the form 
    "Parent --rel--> Child". To build this up, we add rules
    in DFS order, so that the parent nodes have already been added
    to the dict for each child we encounter.

    # Parameters
    dependency_triples: List[List[str]]
        A list of [parent, relation, child] triples, which together
        form a tree that we would like to match on.

    # Returns
    pattern:
        A json structure defining the match for the given tree, which
        can be passed to the spacy DependencyMatcher.

    """
    # Step 1: Build up a dictionary mapping parents to their children
    # in the dependency subtree. Whilst we do this, we check that there is
    # a single node which has only outgoing edges.

    root, parent_to_children = check_for_non_trees(dependency_triples)
    if root is None:
        return None

    def add_node(parent: str, pattern: List):

        for (rel, child) in parent_to_children[parent]:
            # First, we add the specification that we are looking for
            # an edge which connects the child to the parent.
            node = {
                "SPEC": {
                    "NODE_NAME": child,
                    "NBOR_RELOP": ">",
                    "NBOR_NAME": parent
                }
            }
            # We want to match the relation exactly.
            token_pattern = {"DEP": rel}

            # Because we're working specifically with relation extraction
            # in mind, we'll use START_ENTITY and END_ENTITY as dummy
            # placeholders in our list of triples to indicate that we want
            # to match a word which is contained within an entity (or the
            # entity itself if you have added the merge_entities pipe
            # to your pipeline before running the matcher).
            if child not in {"START_ENTITY", "END_ENTITY"}:
                token_pattern["ORTH"] = child
            else:
                token_pattern["ENT_TYPE"] = {"NOT_IN": [""]}

            node["PATTERN"] = token_pattern

            pattern.append(node)
            add_node(child, pattern)

    pattern = [{"SPEC": {"NODE_NAME": root}, "PATTERN": {"ORTH": root}}]
    add_node(root, pattern)

    return pattern

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(merge_entities)

    example = [
        ["founded", "nsubj", "START_ENTITY"],
        ["founded", "dobj", "END_ENTITY"]
    ]

    pattern = construct_pattern(example)
    matcher = DependencyMatcher(nlp.vocab)
    matcher.add("pattern1", None, pattern)

    doc1 = nlp("Bill Gates founded Microsoft.")
    doc2 = nlp("Bill Gates, the Seattle Seahawks owner, founded Microsoft.")

    match = matcher(doc1)[0]
    subtree = match[1][0]
    visualise_subtrees(doc1, subtree)

    match = matcher(doc2)[0]
    subtree = match[1][0]
    visualise_subtrees(doc2, subtree)
