from flask import Flask, request, jsonify
from math import sqrt
import numpy as np

app = Flask(__name__)


def kalman_filter(rssi_list, error_estimate=1, error_measure=2):
    estimate = rssi_list[0]
    for measurement in rssi_list[1:]:
        kalman_gain = error_estimate / (error_estimate + error_measure)
        estimate += kalman_gain * (measurement - estimate)
        error_estimate *= 1 - kalman_gain
    return estimate


def build_fingerprint_db(fingerprint_raw):
    merged = {}

    for fp in fingerprint_raw:
        try:
            x, y = fp["x"], fp["y"]
            beacon_id = fp["beacon_id"]
            rssi_list = fp["rssi_values"]

            if not isinstance(rssi_list, list) or not rssi_list:
                continue

            key = (x, y)
            if key not in merged:
                merged[key] = {}
            merged[key][beacon_id] = kalman_filter(rssi_list)
        except:
            continue

    result = []
    for position, rssis in merged.items():
        result.append({"position": position, "rssis": rssis})

    return result


def knn_localize(filtered_rssi, fingerprint_db, k=3):
    distances = []

    for fp in fingerprint_db:
        common = set(filtered_rssi.keys()) & set(fp["rssis"].keys())
        if not common:
            continue

        dist = sum((filtered_rssi[b] - fp["rssis"][b]) ** 2 for b in common)
        distances.append((fp["position"], sqrt(dist)))

    if not distances:
        return None

    distances.sort(key=lambda x: x[1])
    top_k = distances[:k]

    avg_x = np.mean([p[0][0] for p in top_k])
    avg_y = np.mean([p[0][1] for p in top_k])
    return {"x": avg_x, "y": avg_y}


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()

    if not data or "rssi_map" not in data or "fingerprints" not in data:
        return jsonify({"error": "Missing 'rssi_map' or 'fingerprints'"}), 400

    rssi_map = data["rssi_map"]
    fingerprints = data["fingerprints"]

    if not isinstance(rssi_map, dict) or not isinstance(fingerprints, list):
        return jsonify({"error": "Invalid input format"}), 400

    filtered_rssi = {
        beacon: kalman_filter(values)
        for beacon, values in rssi_map.items()
        if isinstance(values, list) and values
    }

    if not filtered_rssi:
        return jsonify({"error": "Invalid or empty RSSI values"}), 400

    fingerprint_db = build_fingerprint_db(fingerprints)

    position = knn_localize(filtered_rssi, fingerprint_db)

    if not position:
        return jsonify({"error": "Cannot estimate position"}), 400

    return jsonify(position)


if __name__ == "__main__":
    app.run(port=5003, debug=True)
