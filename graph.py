"""Entity graph for OSINT pivoting: subject -> directors, founders, subsidiaries, cases.

Instead of a flat list of leads, the deep search keeps a graph. Every lead the
agent reports becomes a node linked to the entity it was found next to, with the
relation and the source URL on the edge. The next round explores the frontier:
unexplored nodes within MAX_DEPTH hops, the best-supported first (more
independent sources pointing at a link = more likely to be real and relevant).

The graph goes into the report as JSON and as a Mermaid diagram, and every
finding about an affiliate carries its path from the subject
("Субъект → директор → Ахметов Е.Б.").
"""
from dataclasses import asdict, dataclass, field

from models import Subject, SubjectType
from names import person_similarity, skeleton, strip_legal_form

KIND_PRIORITY = {"company": 0, "person": 1, "case": 2, "other": 3}


@dataclass
class Node:
    id: str
    name: str
    kind: str  # subject | company | person | case | other
    depth: int
    explored: bool = False
    sources: set[str] = field(default_factory=set)


@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    source_url: str = ""


def node_key(name: str, kind: str) -> str:
    base = strip_legal_form(name) if kind in ("company", "subject") else name
    return f"{kind}:{skeleton(base)}"


class EntityGraph:
    def __init__(self, subject: Subject):
        self.subject_id = "subject"
        kind = "company" if subject.type == SubjectType.COMPANY else "person"
        self.subject_kind = kind
        self.nodes: dict[str, Node] = {self.subject_id: Node(self.subject_id, subject.name, "subject", 0)}
        self.edges: list[Edge] = []
        # The subject is reachable under its own name and aliases too.
        self._alias: dict[str, str] = {node_key(n, kind): self.subject_id for n in [subject.name, *subject.aliases]}
        self._alias.update({node_key(n, "subject"): self.subject_id for n in [subject.name, *subject.aliases]})

    def find(self, name: str) -> str | None:
        """Node id for a name of any kind (the agent's `connected_to` has no kind)."""
        if not name.strip():
            return None
        for kind in ("subject", "company", "person", "case", "other"):
            key = node_key(name, kind)
            if key in self._alias:
                return self._alias[key]
            if key in self.nodes:
                return key
        # Sources shorten names: "Ахметов Е.Б." is the known director "Ахметов Ержан Болатович".
        best, best_score = None, 0.0
        for node in self.nodes.values():
            if node.kind == "person" or (node.kind == "subject" and self.subject_kind == "person"):
                s, _ = person_similarity(node.name, name)
                if s >= 0.45 and s > best_score:
                    best, best_score = node.id, s
        return best

    def add(self, name: str, kind: str, relation: str, connected_to: str = "", source_url: str = "") -> str:
        """Add (or reinforce) an entity linked to `connected_to` (default: the subject)."""
        parent = self.find(connected_to) or self.subject_id
        existing = self.find(name)
        if existing == self.subject_id:
            return existing
        nid = existing or node_key(name, kind)
        depth = self.nodes[parent].depth + 1
        if nid not in self.nodes:
            self.nodes[nid] = Node(nid, name.strip(), kind, depth)
        node = self.nodes[nid]
        node.depth = min(node.depth, depth)
        if source_url:
            node.sources.add(source_url)
        if not any(e.src == parent and e.dst == nid and e.relation == relation for e in self.edges):
            self.edges.append(Edge(parent, nid, relation, source_url))
        return nid

    def frontier(self, limit: int, max_depth: int) -> list[Node]:
        cands = [n for n in self.nodes.values()
                 if not n.explored and n.id != self.subject_id and n.depth <= max_depth]
        cands.sort(key=lambda n: (-len(n.sources), n.depth, KIND_PRIORITY.get(n.kind, 9), n.name))
        return cands[:limit]

    def mark_explored(self, ids: list[str]) -> None:
        for i in ids:
            if i in self.nodes:
                self.nodes[i].explored = True

    def path(self, nid: str) -> str:
        """Shortest relation path from the subject: 'Субъект → директор: Ахметов Е.Б.'."""
        if nid == self.subject_id or nid not in self.nodes:
            return ""
        prev: dict[str, tuple[str, str]] = {}
        queue, seen = [self.subject_id], {self.subject_id}
        while queue:
            cur = queue.pop(0)
            for e in self.edges:
                if e.src == cur and e.dst not in seen:
                    seen.add(e.dst)
                    prev[e.dst] = (cur, e.relation)
                    queue.append(e.dst)
        steps, cur = [], nid
        while cur in prev:
            src, rel = prev[cur]
            steps.append(f"{rel}: {self.nodes[cur].name}")
            cur = src
        return " → ".join(["Субъект", *reversed(steps)]) if steps else ""

    def to_dict(self) -> dict:
        nodes = []
        for n in self.nodes.values():
            d = asdict(n)
            d["sources"] = sorted(n.sources)
            nodes.append(d)
        return {"nodes": nodes, "edges": [asdict(e) for e in self.edges]}

    def to_mermaid(self, adverse: set[str] | None = None) -> str:
        """Flowchart; nodes with adverse findings are highlighted."""
        adverse = adverse or set()
        ids = {nid: f"n{i}" for i, nid in enumerate(self.nodes)}

        def label(n: Node) -> str:
            text = n.name.replace('"', "'")
            return f'{ids[n.id]}["{text}"]' if n.kind != "case" else f'{ids[n.id]}(["{text}"])'

        lines = ["flowchart LR"]
        lines += [f"  {label(n)}" for n in self.nodes.values()]
        for e in self.edges:
            rel = e.relation.replace('"', "'").replace("|", "/") or "связь"
            lines.append(f'  {ids[e.src]} -->|"{rel}"| {ids[e.dst]}')
        lines.append(f"  style {ids[self.subject_id]} stroke-width:3px")
        lines += [f"  style {ids[nid]} stroke:#c0392b,stroke-width:2px" for nid in adverse if nid in ids]
        return "\n".join(lines)
