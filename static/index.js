let eventsData = [];
let counts = Array(EVENTS.length).fill(0);
let table, barChart, classChart;
let predictionCounts = {}; // cumulative { classLabel: count }

$(document).ready(() => {
  table = $("#event-table").DataTable({
    paging: false,
    info: false,
    searching: false,
    order: [], // keep insertion (chronological) order; timestamps don't sort correctly as text
    columns: [
      { title: "Timestamp", type: "string" }, // step numbers stay left-aligned like times
      { title: "Event" },
      {
        title: "Delete",
        orderable: false,
        defaultContent:
          '<div class="text-center"><button class="btn btn-danger btn-sm delete-btn">X</button></div>',
      },
    ],
  });

  // Bar chart
  const barCtx = $("#barChart")[0].getContext("2d");
  barChart = new Chart(barCtx, {
    type: "bar",
    data: {
      labels: EVENTS,
      datasets: [
        {
          label: "Counts",
          data: counts,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      scales: {
        x: {
          ticks: {
            stepSize: 1,
            callback: function (value) {
              return Number.isInteger(value) ? value : null;
            },
          },
          beginAtZero: true,
        },
      },
    },
  });

  // Classification pie with custom colors
  const classCtx = $("#classChart")[0].getContext("2d");
  classChart = new Chart(classCtx, {
    type: "pie",
    data: {
      labels: [],
      datasets: [
        {
          data: [],
          backgroundColor: ["#4e79a7", "#f28e2b", "#e15759"], // add as many colors as you have classes
        },
      ],
    },
    options: { responsive: true },
  });

  // button toggles
  $(".event-button").click(function () {
    $(this).toggleClass("active");
  });

  $("#update-button").click(async () => {
    const ts = nextTimestamp();
    $(".event-button.active").each(function () {
      const ev = $(this).data("event");
      const idx = EVENTS.indexOf(ev);
      eventsData.push({ time: ts, event: ev });
      table.row.add([ts, ev]).draw();
      counts[idx]++;
      $(this).removeClass("active");
    });
    barChart.data.datasets[0].data = counts;
    barChart.update();

    await refreshClassification();
    await refreshNetwork();
  });

  $("#upload-csv").click(() => $("#upload-csv-input").click());

  $("#upload-csv-input").on("change", async function () {
    const file = this.files[0];
    this.value = ""; // allow re-selecting the same file later
    if (!file) return;
    if (
      eventsData.length > 0 &&
      !confirm("Loading this file will replace the events currently logged. Continue?")
    ) {
      return;
    }
    await loadCsv(file);
  });

  // Delegate click handler for delete buttons
  $("#event-table tbody").on("click", "button.delete-btn", function () {
    // 1. Find the DataTable row
    const row = table.row($(this).closest("tr"));
    const [time, ev] = row.data();

    // 2. Remove from your in-memory eventsData array
    eventsData = eventsData.filter(
      (rec) => !(rec.time === time && rec.event === ev)
    );

    // 3. Decrement the count and update bar chart
    const idx = EVENTS.indexOf(ev);
    if (idx !== -1 && counts[idx] > 0) {
      counts[idx]--;
      barChart.data.datasets[0].data = counts;
      barChart.update();
    }

    // 4. Remove row from table
    row.remove().draw();

    // 5. (Optional) re-run classification & network:
    refreshClassification();
    refreshNetwork();
  });
});

// Sessions loaded from a file without times are numbered 1, 2, 3, ...;
// keep numbering new steps instead of mixing in clock times
function nextTimestamp() {
  const numbered =
    eventsData.length > 0 && eventsData.every((rec) => /^\d+$/.test(rec.time));
  if (numbered) {
    return String(Math.max(...eventsData.map((rec) => Number(rec.time))) + 1);
  }
  return new Date().toLocaleTimeString();
}

async function loadCsv(file) {
  const form = new FormData();
  form.append("file", file);
  let resp, js;
  try {
    resp = await fetch("/upload_csv", { method: "POST", body: form });
    js = await resp.json();
  } catch (err) {
    alert("Could not load the file. Please check your connection and try again.");
    return;
  }
  if (!resp.ok) {
    alert(js.error || "Could not load the file.");
    return;
  }

  // Replace the current session with the uploaded data
  eventsData = js.events;
  counts = Array(EVENTS.length).fill(0);
  table.clear();
  eventsData.forEach((rec) => {
    counts[EVENTS.indexOf(rec.event)]++;
    table.row.add([rec.time, rec.event]);
  });
  table.draw();
  barChart.data.datasets[0].data = counts;
  barChart.update();

  // One prediction per timestamp, as if each was an Update click
  predictionCounts = {};
  js.predictions.forEach((pred) => {
    predictionCounts[pred] = (predictionCounts[pred] || 0) + 1;
  });
  classChart.data.labels = Object.keys(predictionCounts);
  classChart.data.datasets[0].data = Object.values(predictionCounts);
  classChart.update();

  await refreshNetwork();
  if (js.notice) alert(js.notice);
}

async function refreshClassification() {
  let resp = await fetch("/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(counts),
  });
  let js = await resp.json();

  // Increment cumulative prediction count
  const pred = js.prediction;
  if (!(pred in predictionCounts)) {
    predictionCounts[pred] = 0;
  }
  predictionCounts[pred] += 1;

  // Update chart with cumulative values
  classChart.data.labels = Object.keys(predictionCounts);
  classChart.data.datasets[0].data = Object.values(predictionCounts);
  classChart.update();
}

async function refreshNetwork() {
  let resp = await fetch("/network", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(eventsData),
  });
  let js = await resp.json();
  $("#networkGraph").attr("src", "data:image/png;base64," + js.img);
}
