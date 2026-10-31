const state = { data: [], filtered: [], map: null, markers: null, charts: {} };
const $ = (selector) => document.querySelector(selector);

function magnitudeColor(value) {
  if (value >= 6) return "#c9323f";
  if (value >= 4.5) return "#eb7041";
  return "#f3aa52";
}

function formatNumber(value, decimals = 0) {
  return Number(value).toLocaleString("en-US", { maximumFractionDigits: decimals, minimumFractionDigits: decimals });
}

function parseDate(value) {
  return new Date(String(value).replace(" ", "T"));
}

function histogram(values, step, max) {
  const bins = [];
  for (let start = step; start < max; start += step) {
    bins.push({ label: start.toFixed(1), count: values.filter(value => value >= start && value < start + step).length });
  }
  return bins;
}

function groupCount(values, labels, key) {
  const counts = labels.map(() => 0);
  values.forEach(value => {
    const index = key(value);
    if (index >= 0 && index < counts.length) counts[index]++;
  });
  return counts;
}

function initMap() {
  state.map = L.map("earthquakeMap", { preferCanvas: true }).setView([31.5, 85], 5);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19
  }).addTo(state.map);
  state.markers = L.layerGroup().addTo(state.map);
}

function updateMap() {
  state.markers.clearLayers();
  const ranked = [...state.filtered].sort((a, b) => b.magnitude - a.magnitude);
  const events = ranked.slice(0, 2500);
  events.forEach(event => {
    const radius = 3 + Math.max(0, event.magnitude - 3) * 1.15;
    const marker = L.circleMarker([event.latitude, event.longitude], {
      radius, color: "#fff", weight: 0.7, fillColor: magnitudeColor(event.magnitude),
      fillOpacity: 0.72
    });
    marker.bindPopup(`<strong>M ${event.magnitude.toFixed(1)}</strong><br>${event.place || "Unknown location"}<br>${event.depth.toFixed(1)} km deep`);
    marker.addTo(state.markers);
  });
}

function chartDefaults() {
  Chart.defaults.font.family = "DM Sans";
  Chart.defaults.color = "#697684";
  Chart.defaults.plugins.legend.display = false;
}

function makeChart(id, type, labels, data, options = {}) {
  if (state.charts[id]) state.charts[id].destroy();
  state.charts[id] = new Chart($(id), {
    type,
    data: { labels, datasets: [{ data, borderWidth: 2, borderColor: "#e96d3a", backgroundColor: "rgba(233,109,58,.16)", fill: true, tension: .3 }] },
    options: { responsive: true, maintainAspectRatio: false, scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: "rgba(105,118,132,.12)" } } }, ...options }
  });
}

function updateCharts() {
  const magnitudes = histogram(state.filtered.map(d => d.magnitude), 0.5, 8);
  makeChart("#magnitudeChart", "bar", magnitudes.map(d => d.label), magnitudes.map(d => d.count), { scales: { y: { beginAtZero: true, grid: { color: "rgba(105,118,132,.12)" } } } });

  const years = [...new Set(state.filtered.map(d => d.year))].sort((a, b) => a - b);
  const counts = years.map(year => state.filtered.filter(d => d.year === year).length);
  makeChart("#yearlyChart", "line", years, counts);

  const depthBins = [
    ["0–10 km", d => d.depth <= 10], ["11–30 km", d => d.depth > 10 && d.depth <= 30],
    ["31–70 km", d => d.depth > 30 && d.depth <= 70], ["71–150 km", d => d.depth > 70 && d.depth <= 150],
    ["150+ km", d => d.depth > 150]
  ];
  makeChart("#depthChart", "bar", depthBins.map(d => d[0]), depthBins.map(d => state.filtered.filter(d[1]).length), { scales: { y: { beginAtZero: true } } });

  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  makeChart("#monthlyChart", "bar", months, months.map((_, index) => state.filtered.filter(d => d.month === index + 1).length), {
    scales: { y: { beginAtZero: true } }, datasets: null
  });
}

function updateRecent() {
  const recent = [...state.filtered].sort((a, b) => b.date - a.date).slice(0, 8);
  $("#recentEvents").innerHTML = recent.map(event => `
    <div class="recent-item">
      <div class="recent-item-top"><strong>${event.place || "Unknown location"}</strong><span class="magnitude-badge">${event.magnitude.toFixed(1)}</span></div>
      <small>${event.date.toLocaleDateString("en-US", { day: "numeric", month: "short", year: "numeric" })} &nbsp;·&nbsp; ${event.depth.toFixed(1)} km depth</small>
    </div>`).join("");
}

function applyFilters() {
  const magnitude = Number($("#minMagnitude").value);
  const start = $("#startDate").value ? new Date(`${$("#startDate").value}T00:00:00`) : null;
  const end = $("#endDate").value ? new Date(`${$("#endDate").value}T23:59:59`) : null;
  state.filtered = state.data.filter(event =>
    event.magnitude >= magnitude &&
    (!start || event.date >= start) &&
    (!end || event.date <= end)
  );

  $("#magnitudeValue").value = magnitude.toFixed(1);
  $("#totalEvents").textContent = formatNumber(state.filtered.length);
  $("#eventsMeta").textContent = `${formatNumber(state.filtered.length / Math.max(state.data.length, 1) * 100, 1)}% of catalogue`;
  $("#averageMagnitude").textContent = state.filtered.length ? formatNumber(state.filtered.reduce((sum, d) => sum + d.magnitude, 0) / state.filtered.length, 2) : "—";
  const strongest = state.filtered.length ? [...state.filtered].sort((a, b) => b.magnitude - a.magnitude)[0] : null;
  $("#maximumMagnitude").textContent = strongest ? strongest.magnitude.toFixed(1) : "—";
  $("#strongestPlace").textContent = strongest ? strongest.place : "—";
  $("#averageDepth").textContent = state.filtered.length ? `${formatNumber(state.filtered.reduce((sum, d) => sum + d.depth, 0) / state.filtered.length, 1)} km` : "—";
  updateMap();
  updateRecent();
  updateCharts();
}

function loadClusters() {
  Papa.parse("./data/cluster_summary.csv", {
    download: true, header: true, complete({ data }) {
      $("#clusterTable").innerHTML = data.filter(row => row.cluster !== undefined && row.cluster !== "").slice(0, 31).map(row => `
        <tr><td>${row.cluster}</td><td>${formatNumber(row.earthquake_count)}</td><td>${Number(row.average_magnitude).toFixed(2)}</td>
        <td>${Number(row.maximum_magnitude).toFixed(1)}</td><td>${Number(row.average_depth).toFixed(1)} km</td>
        <td>${Number(row.center_latitude).toFixed(2)}, ${Number(row.center_longitude).toFixed(2)}</td></tr>`).join("");
    }
  });
}

function downloadCsv() {
  const rows = ["id,time,place,latitude,longitude,magnitude,depth", ...state.filtered.map(d =>
    [d.id, d.raw.time, d.place, d.latitude, d.longitude, d.magnitude, d.depth].map(value => `"${String(value ?? "").replaceAll('"', '""')}"`).join(",")
  )];
  const url = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a"); link.href = url; link.download = "himalayan-earthquakes.csv"; link.click(); URL.revokeObjectURL(url);
}

window.addEventListener("DOMContentLoaded", () => {
  initMap();
  chartDefaults();
  loadClusters();
  Papa.parse("./data/processed_earthquakes.csv", {
    download: true, header: true, skipEmptyLines: true,
    complete({ data }) {
      state.data = data.map(raw => ({
        id: raw.id, place: raw.place, magnitude: Number(raw.magnitude), depth: Number(raw.depth),
        latitude: Number(raw.latitude), longitude: Number(raw.longitude), year: Number(raw.year),
        month: Number(raw.month), raw, date: parseDate(raw.time)
      })).filter(d => Number.isFinite(d.latitude) && Number.isFinite(d.longitude) && Number.isFinite(d.magnitude) && Number.isFinite(d.depth));
      const dates = state.data.map(d => d.date).filter(d => !Number.isNaN(d.getTime()));
      const minDate = new Date(Math.min(...dates.map(d => d.getTime())));
      const maxDate = new Date(Math.max(...dates.map(d => d.getTime())));
      $("#startDate").value = minDate.toISOString().slice(0, 10);
      $("#startDate").min = $("#startDate").value;
      $("#startDate").max = maxDate.toISOString().slice(0, 10);
      $("#endDate").value = maxDate.toISOString().slice(0, 10);
      $("#endDate").min = minDate.toISOString().slice(0, 10);
      $("#endDate").max = $("#endDate").value;
      applyFilters();
      $("#loadingOverlay").classList.add("hidden");
    },
    error() {
      $("#loadingOverlay").innerHTML = "<p>Unable to load the seismic catalogue.</p>";
    }
  });

  ["minMagnitude", "startDate", "endDate"].forEach(id => $(`#${id}`).addEventListener("input", applyFilters));
  $("#resetFilters").addEventListener("click", () => {
    $("#minMagnitude").value = $("#minMagnitude.min");
    $("#startDate").value = $("#startDate").min;
    $("#endDate").value = $("#endDate").max;
    applyFilters();
  });
  $("#downloadCsv").addEventListener("click", downloadCsv);
  $("#themeToggle").addEventListener("click", () => document.body.classList.toggle("dark"));
});
