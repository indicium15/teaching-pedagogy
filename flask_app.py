from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd, io, base64
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xhtml2pdf import pisa
from datetime import date
import joblib
import networkx as nx
import itertools
import numpy as np
import random

app = Flask(__name__)
# load your trained RF
loaded_rf = joblib.load("./decision_tree/NLP_RFtrained.joblib")
with open("./decision_tree/rf_features.txt") as f:
    RF_FEATURES = [line.strip() for line in f]


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

    # Reset index to make 'time' a column
    pivot_df = pivot_df.reset_index()

    # Reorder columns (some may be missing initially)
    column_order = ['time'] + EVENTS
    pivot_df = pivot_df.reindex(columns=column_order)

    # Fill missing with 0 and cast all except 'time' to int
    for col in pivot_df.columns:
        if col != 'time':
            pivot_df[col] = pivot_df[col].fillna(0).astype(int)

    pivot_df.sort_values(by='time')
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
    times = [datetime.strptime(r['time'], '%I:%M:%S %p') for r in events]

    duration = {'NLP': 0, 'LP': 0}
    if len(events) > 1:
        times = [datetime.strptime(r['time'], '%I:%M:%S %p') for r in events]
        for i in range(len(events)-1):
            counts = [1 if events[i]['event']==e else 0 for e in EVENTS]
            df = pd.DataFrame([counts], columns=RF_FEATURES)
            label = loaded_rf.predict(df)[0]
            duration[label] += (times[i+1]-times[i]).seconds
    else:
        # fallback to equal durations
        duration['NLP'] = 1
        duration['LP'] = 1

    # Teacher pie
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([duration['NLP'], duration['LP']],
        labels=["Linear Pedagogy", "Non-Linear Pedagogy"],
        autopct='%1.1f%%', startangle=90)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Add margin
    buf_t = io.BytesIO(); fig.savefig(buf_t, format='png'); plt.close(fig)
    teacher_img = base64.b64encode(buf_t.getvalue()).decode()

    # Student pie
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([duration['NLP'], duration['LP']],
        labels=[
            f"Linear Pedagogy ({duration['LP']} seconds)",
            f"Non-Linear Pedagogy ({duration['NLP']} seconds)"
        ],
        autopct='%1.1f%%', startangle=90)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Add margin
    buf_s = io.BytesIO(); fig.savefig(buf_s, format='png'); plt.close(fig)
    student_img = base64.b64encode(buf_s.getvalue()).decode()


    # 3) network + TCI as before
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
    return send_file(pdf_buf, mimetype='application/pdf',
                     as_attachment=True, download_name='report.pdf')

if __name__=='__main__':
    app.run(debug=True)
