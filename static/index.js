let eventsData = [];
let counts = Array(EVENTS.length).fill(0);
let table, barChart, classChart;
let predictionCounts = {}; // cumulative { classLabel: count }

$(document).ready(() => {
  table = $("#event-table").DataTable({
    paging: false,
    info: false,
    searching: false,
    columns: [
      { title: "Timestamp" },
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
    const ts = new Date().toLocaleTimeString();
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
