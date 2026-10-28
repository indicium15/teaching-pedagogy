from flask import Flask, render_template, request, jsonify, send_file, Response
import pandas as pd, io, base64
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xhtml2pdf import pisa
from datetime import date, time, timedelta
import joblib
import networkx as nx
import itertools
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # caps uploads (a full lesson CSV is ~50 KB)

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "The file is too large (max 5 MB)."}), 413
# load your trained RF
# PythonAnywhere dev
loaded_rf = joblib.load("/home/jkomar/kto/teaching-pedagogy/decision_tree/NLP_RFtrained.joblib")
with open("/home/jkomar/kto/teaching-pedagogy/decision_tree/rf_features.txt") as f:
    RF_FEATURES = [line.strip() for line in f]

# Local Dev
# loaded_rf = joblib.load("./decision_tree/NLP_RFtrained.joblib")
# with open("./decision_tree/rf_features.txt") as f:
#     RF_FEATURES = [line.strip() for line in f]

EVENTS = [
  "Isolated technical skill practice","Game representative skill","Modified game",
  "Small sided game","Regular play","Repetitive task","Room for variability",
  "Infusion of space variability","Infusion of player variability","Infusion of equipment variability",
  "Prescriptive instruction","Use of analogy","Movement forms",
  "Movement outcomes/effects","Oral promotion of variability",
  "Prescriptive feedback","Feedback via analogy","Feedback on movement forms",
  "Feedback on movement outcomes/effects","Feedback on variability"
]

ABBREV = {
  "Isolated technical skill practice":"IT","Game representative skill":"GRS","Modified game":"MG",
  "Small sided game":"SSG","Regular play":"RP","Repetitive task":"REP","Room for variability":"RFV",
  "Infusion of space variability":"ISV","Infusion of player variability":"IPV","Infusion of equipment variability":"IEV",
  "Prescriptive instruction":"PSI","Use of analogy":"ANA","Movement forms":"MF",
  "Movement outcomes/effects":"MOE","Oral promotion of variability":"OPV",
  "Prescriptive feedback":"PF","Feedback via analogy":"FvA","Feedback on movement forms":"FMF",
  "Feedback on movement outcomes/effects":"FMO","Feedback on variability":"FoV"
}

@app.route('/')
def index():
    return render_template('index.html',
      all_events=EVENTS,
      practice_events=EVENTS[:10],
      instruction_events=EVENTS[10:15],
      feedback_events=EVENTS[15:]
    )

@app.route('/classify', methods=['POST'])
def classify():
    input_data = request.get_json()
    if len(input_data) != len(RF_FEATURES):
        return jsonify({"error": "Invalid input length"}), 400
    df = pd.DataFrame([input_data], columns=RF_FEATURES)
    pred = loaded_rf.predict(df)[0]
    probs = loaded_rf.predict_proba(df)[0]
    return jsonify({
        "prediction": pred,
        "classes": list(loaded_rf.classes_),
        "probs": list(probs)
    })


def build_network(events):
    # events: list of {time, event}
    by_time = {}
    for rec in events:
        by_time.setdefault(rec['time'], []).append(rec['event'])
    edge_w = {}
    for grp in by_time.values():
        for a,b in itertools.permutations(sorted(set(grp)),2):
            edge_w[(a,b)] = edge_w.get((a,b),0) + 1

    # Abbreviate & build graph
    G = nx.Graph()
    for (a,b),w in edge_w.items():
        G.add_edge(ABBREV[a], ABBREV[b], weight=w)

    # ---- TCI calculation (just like reference) ----
    # number of distinct nodes & edges:
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    weights = np.array(list(nx.get_edge_attributes(G,'weight').values()), dtype=float)
    std_w = weights.std() if weights.std() >= 1 else 1.0
    tci = num_nodes * (num_edges / std_w) / 100.0

    return G, tci

@app.route('/network', methods=['POST'])
def network():
    events = request.get_json()
    G, tci = build_network(events)

    fig, ax = plt.subplots(figsize=(6,6))
    pos     = nx.spring_layout(G, k=1)
    weights = [G[u][v]['weight'] for u,v in G.edges()]

    nx.draw(
      G, pos, ax=ax, with_labels=True,
      width=weights, node_color='#66c2a5',
      font_color='white', node_size=500
    )

    # Annotate the TCI on the plot (bottom right, no border)
    fig.text(
        0.95, 0.05, f"TCI: {tci:.2f}",
        ha='right', va='bottom',
        fontsize=12,
        bbox=None
    )
        
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"img": img})


@app.route('/download_csv', methods=['POST'])
def download_csv():
    data = request.get_json()
    df = pd.DataFrame(data)
    if df.empty:
        return "No data", 400

    df['value'] = 1
    pivot_df = df.pivot_table(index='time', columns='event', values='value', fill_value=0)

    # Keep rows in recorded order, then make 'time' a column
    pivot_df = pivot_df.reindex(df['time'].unique()).reset_index()

    # Reorder columns (some may be missing initially)
    column_order = ['time'] + EVENTS
    pivot_df = pivot_df.reindex(columns=column_order)

    # Fill missing with 0 and cast all except 'time' to int
    for col in pivot_df.columns:
        if col != 'time':
            pivot_df[col] = pivot_df[col].fillna(0).astype(int)

    # Save to buffer and send as file
    buf = io.StringIO()
    pivot_df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='activities.csv'
    )

TIME_FORMATS = ['%I:%M:%S %p', '%I:%M %p', '%H:%M:%S', '%H:%M']

def parse_time(value):
    # Excel time/datetime cells arrive as objects rather than text
    if isinstance(value, datetime):
        return datetime.combine(date(1900, 1, 1), value.time())
    if isinstance(value, time):
        return datetime.combine(date(1900, 1, 1), value)
    s = str(value).strip().upper()
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None

def read_table(file):
    if file.filename.lower().endswith('.xlsx'):
        return pd.read_excel(file)
    raw = file.read()
    # Excel on Windows may save as cp1252 rather than UTF-8; latin-1 never fails
    for encoding in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            pass
    # sep=None sniffs the delimiter, e.g. ';' from Excel in European locales
    return pd.read_csv(io.StringIO(text), sep=None, engine='python')

def is_step_number(value):
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value).is_integer() and value >= 0
    return str(value).strip().isdigit()

def to_seconds(dt):
    return dt.hour * 3600 + dt.minute * 60 + dt.second

def format_seconds(seconds):
    # Same format the browser's toLocaleTimeString() produces, e.g. "9:05:03 AM"
    return (datetime(1900, 1, 1) + timedelta(seconds=int(round(seconds)))).strftime('%I:%M:%S %p').lstrip('0')

def resolve_times(times):
    """Turn the time column into sortable positions and display labels.

    An empty time column numbers the rows 1, 2, 3, ... in file order.
    Returns (positions, labels, notice) or raises ValueError with a user-facing message.
    """
    present = times.notna()
    if not present.any():
        positions = pd.Series(range(1, len(times) + 1), index=times.index)
        return positions, positions.astype(str), (
            "This file has no times, so its rows were numbered 1, 2, 3, … in file order.")
    if not present.all():
        row = times.index[~present][0] + 2  # +2: header line and 1-based rows
        raise ValueError(f"Row {row} has no time. Fill in every time, or leave the whole "
                         "time column empty to number the rows 1, 2, 3, … automatically.")

    # Step numbers, e.g. from re-uploading a numbered session the app downloaded
    if all(is_step_number(v) for v in times):
        positions = times.astype(float).astype(int)
        return positions, positions.astype(str), None
    if any(is_step_number(v) for v in times):
        raise ValueError("The time column mixes step numbers and clock times. Please use one or the other.")

    parsed = times.map(parse_time)
    if parsed.isna().any():
        raise ValueError(f"Unrecognised time value: '{times[parsed.isna()].iloc[0]}'.")
    seconds = parsed.map(to_seconds)
    return seconds, seconds.map(format_seconds), None

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({"error": "No file uploaded."}), 400
    try:
        df = read_table(file)
    except Exception:
        return jsonify({"error": "Could not read the file. Please upload a CSV or Excel (.xlsx) file."}), 400

    # Match headers ignoring case and surrounding spaces
    canonical = {name.lower(): name for name in ['time'] + EVENTS}
    df.columns = [canonical.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
    if 'time' not in df.columns:
        df['time'] = None
    event_cols = [e for e in EVENTS if e in df.columns]
    if not event_cols:
        return jsonify({"error": "The file has no recognised event columns."}), 400

    df = df.dropna(subset=['time'] + event_cols, how='all')  # blank rows
    df['time'] = df['time'].where(df['time'].astype(str).str.strip() != '')
    try:
        df['position'], df['label'], notice = resolve_times(df['time'])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    values = df.reindex(columns=EVENTS).apply(pd.to_numeric, errors='coerce')
    if values[event_cols].isna().any().any():
        return jsonify({"error": "Event columns must contain only numbers (0 or 1)."}), 400
    df[EVENTS] = (values.fillna(0) > 0).astype(int)

    # One "Update" step per distinct time (or step number), in order
    steps = df.groupby(['position', 'label'], sort=True)[EVENTS].max()
    steps = steps[steps.sum(axis=1) > 0]
    if steps.empty:
        return jsonify({"error": "The file contains no recorded events."}), 400

    events = []
    for (_, label), row in steps.iterrows():
        events.extend({"time": label, "event": e} for e in EVENTS if row[e])

    # Classify the cumulative counts after each step
    cumulative = pd.DataFrame(steps.cumsum().to_numpy(), columns=RF_FEATURES)
    predictions = loaded_rf.predict(cumulative)
    return jsonify({"events": events, "predictions": [str(p) for p in predictions], "notice": notice})

def event_positions(labels):
    """Positions used to measure how long each event lasted, and the unit for the PDF."""
    if all(is_step_number(l) for l in labels):
        return [int(l) for l in labels], " steps"
    parsed = [parse_time(l) for l in labels]
    if all(p is not None for p in parsed):
        return [to_seconds(p) % 86400 for p in parsed], "s"
    # Mixed or unknown labels: treat each distinct label as one step
    order = {l: i for i, l in enumerate(dict.fromkeys(labels))}
    return [order[l] for l in labels], " steps"

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    events = request.get_json()

    # 1) frequency horizontal bar
    df_events = pd.DataFrame(events)
    freq = df_events['event'].value_counts().reindex(EVENTS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 8))
    freq.plot.barh(ax=ax)
    ax.set_xlabel("Count"); ax.set_ylabel("Event")
    fig.tight_layout()
    buf_freq = io.BytesIO(); fig.savefig(buf_freq,format='png'); plt.close(fig)
    freq_img = base64.b64encode(buf_freq.getvalue()).decode()

    # 2) compute durations & two pies
    positions, unit = event_positions([r['time'] for r in events])

    duration = {'NLP': 0, 'LP': 0}
    if len(events) > 1:
        for i in range(len(events)-1):
            counts = [1 if events[i]['event']==e else 0 for e in EVENTS]
            df = pd.DataFrame([counts], columns=RF_FEATURES)
            label = loaded_rf.predict(df)[0]
            gap = positions[i+1]-positions[i]
            duration[label] += gap % 86400 if unit == "s" else gap  # clock times may cross midnight
    else:
        # fallback to equal durations
        duration['NLP'] = 1
        duration['LP'] = 1

    # Teacher pie
    fig, ax = plt.subplots(figsize=(5, 5))
    # Ensure we have non-zero values to plot
    if duration['NLP'] == 0 and duration['LP'] == 0:
        # If both are 0, set them to 1 for equal representation
        plot_values = [1, 1]
    else:
        plot_values = [duration['NLP'], duration['LP']]
    ax.pie(
        plot_values,
        labels=["Linear Pedagogy", "Non-Linear Pedagogy"],
        autopct="%1.1f%%",
        startangle=90,
        labeldistance=1.1,
        pctdistance=0.85
    )
    ax.axis("equal")
    fig.tight_layout()
    buf_t = io.BytesIO()
    fig.savefig(
        buf_t,
        format="png",
        bbox_inches="tight",
        pad_inches=0.2
    )
    plt.close(fig)
    buf_t.seek(0)
    teacher_img = base64.b64encode(buf_t.getvalue()).decode()

    # Student pie
    fig, ax = plt.subplots(figsize=(5, 5))
    # Ensure we have non-zero values to plot
    if duration['NLP'] == 0 and duration['LP'] == 0:
        # If both are 0, set them to 1 for equal representation
        plot_values = [1, 1]
    else:
        plot_values = [duration['NLP'], duration['LP']]
    ax.pie(
        plot_values,
        labels=[
            f"Linear Pedagogy ({duration['LP']}{unit})",
            f"Non-Linear Pedagogy ({duration['NLP']}{unit})"
        ],
        autopct="%1.1f%%",
        startangle=90,
        labeldistance=1.1,
        pctdistance=0.85
    )
    ax.axis("equal")
    fig.tight_layout()
    buf_s = io.BytesIO()
    fig.savefig(
        buf_s,
        format="png",
        bbox_inches="tight",
        pad_inches=0.2
    )
    plt.close(fig)
    buf_s.seek(0)
    student_img = base64.b64encode(buf_s.getvalue()).decode()

    # 3) network + TCI
    G, tci = build_network(events)
    fig, ax = plt.subplots(figsize=(5,5))
    pos = nx.spring_layout(G, k=1)
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    nx.draw(G,pos,ax=ax,with_labels=True,
            width=weights,node_color='#fc8d62',
            font_color='white',node_size=400)
    fig.text(
        0.95, 0.05, f"TCI: {tci:.2f}",
        ha='right', va='bottom',
        fontsize=12,
        bbox=None
    )
    fig.tight_layout()
    buf_net = io.BytesIO(); fig.savefig(buf_net,format='png'); plt.close(fig)
    net_img = base64.b64encode(buf_net.getvalue()).decode()

    # render PDF
    html = render_template('report.html',
                           today=date.today().isoformat(),
                           freq_img=freq_img,
                           teacher_img=teacher_img,
                           student_img=student_img,
                           net_img=net_img)
    pdf_buf = io.BytesIO()
    pisa.CreatePDF(html, dest=pdf_buf)
    pdf_buf.seek(0)
    return Response(
        pdf_buf,
        mimetype='application/pdf',
        headers={
            'Content-Disposition': 'attachment; filename=report.pdf'
        }
    )

if __name__=='__main__':
    app.run(debug=True)
