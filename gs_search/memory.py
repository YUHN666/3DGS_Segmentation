"""Conservative RGB feature association using known camera poses.

No box-center triangulation, semantic map or rendered depth. Actual SIFT pixel
correspondences must pass reciprocal ratio tests, epipolar geometry, cheirality,
reprojection, parallax and spatial support checks. Ambiguity stays pending.
Triangulated feature points are internal association checks, NOT target positions.
"""
from dataclasses import dataclass, field
import hashlib
import cv2
import numpy as np


def evidence_quality(e,config):
    x1,y1,x2,y2=e.detection.box
    w,h=e.camera.width,e.camera.height
    margin=config.get('box_border_margin',4)
    reasons=[]
    if min(x1,y1,w-x2,h-y2)<margin: reasons.append('clipped_box')
    if (x2-x1)*(y2-y1)/(w*h)>config.get('max_box_area_fraction',.75): reasons.append('oversized_box')
    if min(x2-x1,y2-y1)<config.get('min_box_side',16): reasons.append('too_small_box')
    return dict(usable=not reasons,reasons=reasons)


def review_panel(e,number):
    """One RGB panel per real view: marked context plus the candidate crop."""
    from PIL import Image,ImageDraw
    full=Image.fromarray(e.rgb)
    draw=ImageDraw.Draw(full)
    draw.rectangle(e.detection.box,outline='red',width=3)
    x1,y1,x2,y2=e.detection.box
    crop=Image.fromarray(e.rgb).crop((max(0,x1-4),max(0,y1-4),min(e.camera.width,x2+4),min(e.camera.height,y2+4)))
    scale=min(320/max(1,crop.width),e.camera.height/max(1,crop.height))
    crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS)
    panel=Image.new('RGB',(e.camera.width+320,e.camera.height+28),(20,25,35))
    panel.paste(full,(0,28));panel.paste(crop,(e.camera.width+(320-crop.width)//2,28+(e.camera.height-crop.height)//2))
    ImageDraw.Draw(panel).text((8,7),f'View {number}: full context (red box) | same-object crop',fill='white')
    return np.asarray(panel)


@dataclass
class Evidence:
    observation_id: int
    camera: object
    detection: object
    rgb: np.ndarray
    keypoints: np.ndarray
    descriptors: object
    image_hash: str


def evidence(obs_id, camera, detection, rgb):
    mask = np.zeros(rgb.shape[:2], np.uint8)
    x1, y1, x2, y2 = np.array(detection.box).astype(int)
    mask[max(y1, 0):y2, max(x1, 0):x2] = 255
    k, d = cv2.SIFT_create(nfeatures=1400, contrastThreshold=.015).detectAndCompute(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), mask)
    xy = np.array([p.pt for p in k], np.float32).reshape(-1, 2)
    return Evidence(obs_id, camera, detection, rgb, xy, d, hashlib.sha256(rgb.tobytes()).hexdigest())


def associate(a, b, config):
    base = float(np.linalg.norm(a.camera.position - b.camera.position))
    result = dict(accepted=False, baseline=base, matches=0, inliers=0)
    if a.image_hash == b.image_hash:
        return dict(result, reason="duplicate_rgb")
    if base < config.get("min_baseline", .3):
        return dict(result, reason="insufficient_translation")
    if a.descriptors is None or b.descriptors is None or min(len(a.descriptors), len(b.descriptors)) < 8:
        return dict(result, reason="insufficient_rgb_features")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    ab, ba = matcher.knnMatch(a.descriptors, b.descriptors, k=2), matcher.knnMatch(b.descriptors, a.descriptors, k=2)
    ratio = config.get("ratio_test", .7)
    reverse = {m.queryIdx: m.trainIdx for pair in ba if len(pair) == 2 for m, n in [pair] if m.distance < ratio*n.distance}
    matches = [m for pair in ab if len(pair) == 2 for m, n in [pair]
               if m.distance < ratio*n.distance and reverse.get(m.trainIdx) == m.queryIdx]
    result["matches"] = len(matches)
    minimum = config.get("min_feature_inliers", 10)
    if len(matches) < minimum:
        return dict(result, reason="insufficient_reciprocal_matches")
    pa = a.keypoints[[m.queryIdx for m in matches]].astype(float)
    pb = b.keypoints[[m.trainIdx for m in matches]].astype(float)
    wa, wb = np.linalg.inv(a.camera.c2w), np.linalg.inv(b.camera.c2w)
    rel = wb @ a.camera.c2w
    t = rel[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(b.camera.K).T @ tx @ rel[:3, :3] @ np.linalg.inv(a.camera.K)
    ah, bh = np.c_[pa, np.ones(len(pa))], np.c_[pb, np.ones(len(pb))]
    la, lb = bh @ F, ah @ F.T
    residual = (bh * lb).sum(1)**2
    sampson = residual / np.maximum((la[:, :2]**2).sum(1) + (lb[:, :2]**2).sum(1), 1e-20)
    P1, P2 = a.camera.K @ wa[:3], b.camera.K @ wb[:3]
    Xh = cv2.triangulatePoints(P1, P2, pa.T, pb.T).T
    valid_w = np.abs(Xh[:, 3]) > 1e-10
    X = Xh[:, :3] / np.where(valid_w, Xh[:, 3], 1)[:, None]
    q1, q2 = np.c_[X, np.ones(len(X))] @ P1.T, np.c_[X, np.ones(len(X))] @ P2.T
    uv1 = q1[:, :2] / np.maximum(q1[:, 2:], 1e-12)
    uv2 = q2[:, :2] / np.maximum(q2[:, 2:], 1e-12)
    ray1, ray2 = X - a.camera.position, X - b.camera.position
    cos = (ray1*ray2).sum(1) / np.maximum(np.linalg.norm(ray1, axis=1)*np.linalg.norm(ray2, axis=1), 1e-12)
    angles = np.degrees(np.arccos(cos.clip(-1, 1)))
    inliers = (valid_w & (q1[:, 2] > 0) & (q2[:, 2] > 0) & (sampson < 2.25) &
               (np.linalg.norm(uv1-pa, axis=1) < 2.5) & (np.linalg.norm(uv2-pb, axis=1) < 2.5) &
               (angles >= config.get("min_parallax_degrees", .7)))
    result["inliers"] = int(inliers.sum())
    result["inlier_fraction"] = float(inliers.mean())
    result["median_parallax_degrees"] = float(np.median(angles[inliers])) if inliers.any() else 0
    if inliers.sum() < minimum or inliers.mean() < .6:
        return dict(result, reason="pose_or_parallax_inconsistent")
    for points, ev in [(pa[inliers], a), (pb[inliers], b)]:
        box = np.array(ev.detection.box)
        span = np.ptp(points, axis=0) / np.maximum(box[2:]-box[:2], 1)
        if np.prod(span) < config.get("min_feature_box_fraction", .08):
            return dict(result, reason="features_concentrated_in_small_box_region")
    return dict(result, accepted=True, reason="rgb_correspondences_pose_consistent")


@dataclass
class Candidate:
    id: str
    sightings: list = field(default_factory=list)
    links: list = field(default_factory=list)
    status: str = "pending"
    verify_attempts: int = 0
    semantic_review: dict = field(default_factory=dict)
    review_history: list = field(default_factory=list)
    confirmation_observation_ids: list = field(default_factory=list)

    def summary(self):
        return dict(id=self.id, status=self.status, verify_attempts=self.verify_attempts,
                    evidence=[dict(observation_id=e.observation_id, box=e.detection.box,
                                   score=e.detection.score, label=e.detection.label,
                                   camera=e.camera.as_dict()) for e in self.sightings],
                    association_links=self.links, semantic_review=self.semantic_review,
                    review_history=self.review_history,confirmation_observation_ids=self.confirmation_observation_ids,
                    target_xyz=None)


class ObservationMemory:
    def __init__(self, config):
        self.config, self.candidates, self.observations = config, [], []
        if config.get("confirmation_views", 2) < 2:
            raise ValueError("confirmation_views must be >= 2")
        if config.get("min_baseline", .3) <= 0:
            raise ValueError("min_baseline must be positive")
        if config.get("min_feature_inliers", 10) < 8:
            raise ValueError("min_feature_inliers must be >= 8")

    def ingest(self, obs_id, camera, rgb, detections):
        self.observations.append(dict(id=obs_id, position=camera.position.tolist(),
                                      candidate_count=len(detections)))
        evs = [evidence(obs_id, camera, d, rgb) for d in detections]
        proposals, diagnostics = [], []
        for e in evs:
            compatible = []
            for c in self.candidates:
                if any(s.observation_id == obs_id for s in c.sightings):
                    continue
                # Require association to anchor; do not accumulate an unverified
                # chain of short-baseline frame tracking as cross-view evidence.
                result = associate(c.sightings[0], e, self.config)
                diagnostics.append(dict(candidate_id=c.id, detection_box=e.detection.box, **result))
                if result["accepted"]:
                    compatible.append((c, result))
            proposals.append((e, compatible))
        # Enforce uniqueness in both directions. Similar instances do not merge
        # because of category, box proximity or appearance alone.
        used = [c.id for _, matches in proposals for c, _ in matches]
        touched = []
        for ev, matches in proposals:
            if len(matches) == 1 and used.count(matches[0][0].id) == 1:
                c, link = matches[0]
                c.sightings.append(ev)
                c.links.append(dict(from_observation=c.sightings[0].observation_id,
                                    to_observation=obs_id, **link))
            else:
                c = Candidate(f"candidate_{len(self.candidates):04d}", [ev])
                if matches:
                    c.status = "pending_ambiguous_association"
                self.candidates.append(c)
            touched.append(c)
        return touched, diagnostics

    def confirm(self, candidate, verifier, target_text, require_full_description=False):
        if candidate.status.startswith(('rejected','confirmed')):
            return False
        eligible=[e for e in candidate.sightings if evidence_quality(e,self.config)['usable']]
        required=self.config.get('confirmation_views',2)
        if len(eligible)<required:
            if any(not evidence_quality(e,self.config)['usable'] for e in candidate.sightings):
                candidate.status='pending_wider_view'
            return False
        # Each additional view must have a validated anchor link and baseline.
        if len(candidate.links) < self.config.get("confirmation_views", 2)-1:
            return False
        if len({e.image_hash for e in candidate.sightings}) < self.config.get("confirmation_views", 2):
            return False
        # The original anchor may have been clipped. Require a valid association
        # among the actual, sufficiently complete views used for confirmation.
        group=[]
        for anchor in eligible[-6:]:
            others=[e for e in eligible[-6:] if e is not anchor and associate(anchor,e,self.config)['accepted']]
            if len(others)>=required-1:
                group=[anchor,*others[:required-1]]
                break
        if not group:
            candidate.status='pending_rgb_identity'
            return False
        ids=sorted(e.observation_id for e in group)
        if verifier.mode != "disabled":
            # Mark the associated instance in each full frame, preserving the
            # context needed for relations. An unrelated matching object
            # elsewhere in the frame must not satisfy this candidate's review.
            if candidate.semantic_review.get('observation_ids')==ids:
                return False
            marked=[review_panel(e,i+1) for i,e in enumerate(group)]
            candidate.semantic_review = dict(verifier.verify(marked,target_text),observation_ids=ids)
            candidate.review_history.append(dict(candidate.semantic_review))
            if candidate.semantic_review.get("matches_target") is not True:
                candidate.status = ('rejected_semantic' if candidate.semantic_review.get('decision')=='reject'
                                    else 'pending_semantic_review')
                return False
        elif require_full_description:
            candidate.status = "pending_full_description_requires_vlm"
            return False
        else:
            candidate.semantic_review = verifier.verify([], target_text)
        candidate.status = "confirmed_category" if verifier.mode == "disabled" else "confirmed_with_vlm"
        candidate.confirmation_observation_ids=ids
        return True
