"""Follow everyone in view: an anonymous id per person, so each gets their own observations.

PeopleTracker gives each pose a person_id ("p1", "p2", ...) that stays with that person from
frame to frame: poses are matched to recent people so the total movement of box centres is
smallest, no pair further apart than max_dist; a pose nobody claims starts a new person. Ids
are never reused in a session. They are anonymous tracks, not identities: two people crossing
or someone leaving and coming back can end up with a new id.

Each person keeps their own hip-velocity estimate and briefly-held joints, so their posture
numbers are worked out from their own history. A person missing for up to keep_s is a brief
dropout (nothing is sent for them, and they keep their id: poseNet often loses someone lying
down for a few frames, and a new id would restart the backend's floor timer). Missing longer
than that, they have ended: the edge sends one "not detected" observation for their id, then
forgets them. keep_s matches the backend's max_observation_gap_seconds (1 s).
"""

from hearth_edge import features

MAX_EXACT = 6   # up to this many people (either side), find the best matching; beyond, greedy


class Person(object):
    def __init__(self, person_id, pose, t, alpha, reset_gap_s, hold_s):
        self.person_id = person_id
        self.center = features.bbox_center(pose["bbox"])
        self.t = t
        self.velocity = features.VerticalVelocity(alpha=alpha, reset_gap_s=reset_gap_s)
        self.hold = features.KeypointHold(hold_s)


class PeopleTracker(object):
    def __init__(self, max_dist, keep_s=1.0, alpha=0.5, reset_gap_s=1.0, hold_s=0.3):
        self.max_dist = max_dist      # pixels between box centres on consecutive frames
        self.keep_s = keep_s
        self.alpha, self.reset_gap_s, self.hold_s = alpha, reset_gap_s, hold_s
        self.people = []              # Person, in the order they were first seen
        self.next_id = 1

    def assign(self, poses, t):
        """poses (pixel bbox each) -> [(Person, pose)] in the same order as poses."""
        owner = dict((j, self.people[i]) for i, j in self._match(poses))
        out = []
        for j, pose in enumerate(poses):
            person = owner.get(j)
            if person is None:
                person = Person("p%d" % self.next_id, pose, t, self.alpha, self.reset_gap_s, self.hold_s)
                self.next_id += 1
                self.people.append(person)
            person.center, person.t = features.bbox_center(pose["bbox"]), t
            out.append((person, pose))
        return out

    def _match(self, poses):
        """[(person index, pose index)] with the smallest total distance; a pair further apart
        than max_dist is never matched (it costs max_dist to leave either side unmatched)."""
        d = []
        for p in self.people:
            row = []
            for pose in poses:
                x, y = features.bbox_center(pose["bbox"])
                row.append(((x - p.center[0]) ** 2 + (y - p.center[1]) ** 2) ** 0.5)
            d.append(row)
        if len(self.people) > MAX_EXACT or len(poses) > MAX_EXACT:
            return self._greedy(d)
        best = [None, []]

        def search(i, used, cost, pairs):
            if best[0] is not None and cost >= best[0]:
                return
            if i == len(self.people):
                best[0], best[1] = cost, list(pairs)
                return
            for j in range(len(poses)):
                if j not in used and d[i][j] <= self.max_dist:
                    used.add(j)
                    pairs.append((i, j))
                    search(i + 1, used, cost + d[i][j], pairs)
                    pairs.pop()
                    used.discard(j)
            search(i + 1, used, cost + self.max_dist, pairs)   # person i not in this frame

        search(0, set(), 0.0, [])
        return best[1]

    def _greedy(self, d):
        """Closest pairs first: for crowds too big to search every matching."""
        pairs = sorted((d[i][j], i, j) for i in range(len(d)) for j in range(len(d[i])))
        taken_i, taken_j, out = set(), set(), []
        for dist, i, j in pairs:
            if dist > self.max_dist:
                break
            if i not in taken_i and j not in taken_j:
                taken_i.add(i)
                taken_j.add(j)
                out.append((i, j))
        return out

    def missing(self, t):
        """People not seen this frame but only briefly (don't report them as gone yet)."""
        return [p for p in self.people if 0 < t - p.t <= self.keep_s]

    def ended(self, t):
        """People gone longer than keep_s: removed, and returned so each can be reported once."""
        gone = [p for p in self.people if t - p.t > self.keep_s]
        if gone:
            self.people = [p for p in self.people if t - p.t <= self.keep_s]
        return gone
