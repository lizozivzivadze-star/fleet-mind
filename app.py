"""
Fleet Mind — Flask Backend API
================================
Deployment: Render.com (free tier)

Environment variables to set in Render dashboard:
    DATABASE_URL   — auto-set by Render PostgreSQL add-on
    ANTHROPIC_API_KEY — your Anthropic key for AI narration
    SECRET_KEY     — any random string for sessions

Local dev:
    pip install flask flask-cors flask-sqlalchemy psycopg2-binary anthropic
    python backend/app.py
"""

import os, sys, json, time
from datetime import datetime, timezone

from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app, origins="*")

# ── Database ─────────────────────────────────────────────────────────
raw_db_url = os.environ.get("DATABASE_URL", "sqlite:///fleet_mind.db")
# Render Postgres URLs start with "postgres://" — SQLAlchemy needs "postgresql://"
if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"]        = raw_db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"]                     = os.environ.get("SECRET_KEY", "dev-secret-change-me")

db = SQLAlchemy(app)


# ────────────────────────────────────────────────
# MODELS
# ────────────────────────────────────────────────
class MissionScore(db.Model):
    __tablename__ = "mission_scores"

    id                   = db.Column(db.Integer, primary_key=True)
    player_name          = db.Column(db.String(32), nullable=False, default="ANONYMOUS")
    coverage_pct         = db.Column(db.Float,   default=0.0)
    threats_neutralized  = db.Column(db.Integer, default=0)
    total_reward         = db.Column(db.Float,   default=0.0)
    mission_mode         = db.Column(db.String(20), default="training")
    episode              = db.Column(db.Integer, default=1)
    created_at           = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id":                  self.id,
            "player_name":         self.player_name,
            "coverage_pct":        round(self.coverage_pct, 1),
            "threats_neutralized": self.threats_neutralized,
            "total_reward":        round(self.total_reward, 2),
            "mission_mode":        self.mission_mode,
            "episode":             self.episode,
            "created_at":          self.created_at.isoformat() if self.created_at else None,
        }


class TrainingStat(db.Model):
    __tablename__ = "training_stats"

    id           = db.Column(db.Integer, primary_key=True)
    episode      = db.Column(db.Integer)
    mean_reward  = db.Column(db.Float)
    mean_coverage = db.Column(db.Float)
    mean_threats = db.Column(db.Float)
    timesteps    = db.Column(db.Integer)
    recorded_at  = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


# ────────────────────────────────────────────────
# HEALTH
# ────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({
        "status": "online",
        "game":   "Fleet Mind",
        "time":   datetime.now(timezone.utc).isoformat(),
    })


# ────────────────────────────────────────────────
# SCORES
# ────────────────────────────────────────────────
@app.route("/api/scores/leaderboard")
def leaderboard():
    """Top 10 scores by total_reward."""
    mode   = request.args.get("mode")          # optional filter
    limit  = min(int(request.args.get("limit", 10)), 50)

    q = MissionScore.query
    if mode:
        q = q.filter_by(mission_mode=mode)
    rows = q.order_by(MissionScore.total_reward.desc()).limit(limit).all()

    return jsonify([
        {**r.to_dict(), "rank": i + 1}
        for i, r in enumerate(rows)
    ])


@app.route("/api/scores/submit", methods=["POST"])
def submit_score():
    data = request.get_json(silent=True) or {}

    # Basic validation
    player = str(data.get("player_name", "ANONYMOUS"))[:32].upper()
    if not player:
        player = "ANONYMOUS"

    score = MissionScore(
        player_name         = player,
        coverage_pct        = float(data.get("coverage_pct",  0)),
        threats_neutralized = int(data.get("threats_neutralized", 0)),
        total_reward        = float(data.get("total_reward",  0)),
        mission_mode        = str(data.get("mode",  "training"))[:20],
        episode             = int(data.get("episode", 1)),
    )
    db.session.add(score)
    db.session.commit()

    return jsonify({"success": True, "id": score.id, "player": player}), 201


@app.route("/api/scores/recent")
def recent_scores():
    limit = min(int(request.args.get("limit", 20)), 100)
    rows  = MissionScore.query.order_by(
        MissionScore.created_at.desc()
    ).limit(limit).all()
    return jsonify([r.to_dict() for r in rows])


# ────────────────────────────────────────────────
# REPLAY
# ────────────────────────────────────────────────
@app.route("/api/replay/latest")
def get_latest_replay():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "logs", "latest_replay.json")

    if not os.path.exists(path):
        return jsonify({"error": "No replay available. Run training first."}), 404

    with open(path) as f:
        data = json.load(f)

    ep_idx = int(request.args.get("episode", 0))
    replays = data.get("replays", [])
    scores  = data.get("scores",  [])

    if ep_idx >= len(replays):
        ep_idx = 0

    return jsonify({
        "frames":        replays[ep_idx] if replays else [],
        "score":         scores[ep_idx]  if scores  else {},
        "total_episodes": len(replays),
    })


# ────────────────────────────────────────────────
# TRAINING STATS
# ────────────────────────────────────────────────
@app.route("/api/training/stats")
def training_stats():
    """Read latest stats from the training log file."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "logs", "training_log.json")

    if not os.path.exists(path):
        return jsonify({
            "episode": 0, "mean_reward": 0,
            "mean_coverage": 0, "mean_threats": 0,
            "timesteps": 0,
        })

    with open(path) as f:
        data = json.load(f)

    # Persist to DB for history charts
    stat = TrainingStat(
        episode       = data.get("episode",       0),
        mean_reward   = data.get("mean_reward",   0),
        mean_coverage = data.get("mean_coverage", 0),
        mean_threats  = data.get("mean_threats",  0),
        timesteps     = data.get("timesteps",     0),
    )
    db.session.add(stat)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()

    return jsonify(data)


@app.route("/api/training/history")
def training_history():
    """Return last N training stat snapshots for chart rendering."""
    limit = min(int(request.args.get("limit", 100)), 500)
    rows  = TrainingStat.query.order_by(
        TrainingStat.recorded_at.desc()
    ).limit(limit).all()

    rows.reverse()
    return jsonify([{
        "episode":       r.episode,
        "mean_reward":   r.mean_reward,
        "mean_coverage": r.mean_coverage,
        "mean_threats":  r.mean_threats,
        "timesteps":     r.timesteps,
    } for r in rows])


# ────────────────────────────────────────────────
# MISSION CONFIG
# ────────────────────────────────────────────────
@app.route("/api/mission/new", methods=["POST"])
def new_mission():
    """Generate a new randomised mission config for the frontend."""
    import random

    difficulty = request.get_json(silent=True) or {}
    diff = difficulty.get("difficulty", "normal")   # easy | normal | hard

    n_threats = {"easy": 2, "normal": 3, "hard": 4}.get(diff, 3)
    threats = [
        {
            "x": random.randint(200, 460),
            "y": random.randint(30,  470),
            "active": True,
        }
        for _ in range(n_threats)
    ]

    return jsonify({
        "mission_id": random.randint(1000, 9999),
        "difficulty": diff,
        "grid_size":  500,
        "threats":    threats,
        "usv1_start": {"x": 80, "y": 190, "heading": 0},
        "usv2_start": {"x": 80, "y": 310, "heading": 0},
        "target_coverage": 80,
        "time_limit_steps": 500,
    })


# ────────────────────────────────────────────────
# AI NARRATION  (Anthropic Claude)
# ────────────────────────────────────────────────
@app.route("/api/narration", methods=["POST"])
def narration():
    """
    Generate live tactical narration using Claude.
    POST body: { event, coverage_pct, threats_neutralized, step, formation_dist }
    """
    try:
        import anthropic
    except ImportError:
        return jsonify({"text": "AI narration unavailable — anthropic package not installed."}), 200

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"text": "AI narration offline — no API key configured."}), 200

    data    = request.get_json(silent=True) or {}
    event   = data.get("event",   "step")
    cov     = data.get("coverage_pct",        0)
    threats = data.get("threats_neutralized", 0)
    step    = data.get("step",                0)
    form    = data.get("formation_dist",      0)

    prompt = f"""You are a naval AI tactical system providing real-time commentary 
for a dual-USV coordinated patrol simulation.

Current mission state:
- Event: {event}
- Step: {step}/500
- Area covered: {cov:.1f}%
- Threats neutralized: {threats}/3
- Formation distance: {form:.0f}m (optimal: 50-150m)

Write ONE sentence of crisp, military-style tactical commentary about this moment.
Use the event type to guide tone:
- "mission_start" → acknowledge deployment
- "threat_neutralized" → confirm kill
- "formation_break" → warn about dispersal
- "coverage_milestone" → report progress
- "mission_complete" → issue debrief summary

Keep it under 20 words. No markdown. Military radio style."""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 80,
        messages   = [{"role": "user", "content": prompt}]
    )

    text = message.content[0].text.strip()
    return jsonify({"text": text})


# ────────────────────────────────────────────────
# DEBRIEF  (longer Claude summary)
# ────────────────────────────────────────────────
@app.route("/api/debrief", methods=["POST"])
def debrief():
    try:
        import anthropic
    except ImportError:
        return jsonify({"text": "Debrief unavailable."}), 200

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"text": "AI debrief offline."}), 200

    data    = request.get_json(silent=True) or {}
    cov     = data.get("coverage_pct",        0)
    threats = data.get("threats_neutralized", 0)
    reward  = data.get("total_reward",        0)
    steps   = data.get("steps",             500)
    mode    = data.get("mode",         "training")

    prompt = f"""You are a naval AI commander issuing a post-mission debrief report.

Mission statistics:
- Area coverage: {cov:.1f}% (target: 80%)
- Threats neutralized: {threats}/3
- Total RL reward: {reward:.2f}
- Steps used: {steps}/500
- Mode: {mode}

Write a 2-3 sentence debrief in military style. Assess performance, identify 
weaknesses, and give one specific recommendation for the next mission. 
Be direct and precise. No bullet points."""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 150,
        messages   = [{"role": "user", "content": prompt}]
    )

    return jsonify({"text": message.content[0].text.strip()})


# ────────────────────────────────────────────────
# STARTUP
# ────────────────────────────────────────────────
def create_tables():
    with app.app_context():
        db.create_all()
        print("✓ Database tables ready")


if __name__ == "__main__":
    create_tables()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    print(f"\n◈ Fleet Mind API running on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
