import asyncio
import threading
from collections import deque

import numpy as np
import requests
from bleak import AdvertisementData, BleakClient, BleakScanner, BLEDevice
from flask import Flask, request

app = Flask(__name__)
should_send_data = False
current_uuid = None
target_ble_names = []
active_monitors = {}
rssi_history = {}
kalman_filters = {}

SEEABLE_PREFIX = "seeable-"
TARGET_BLE_NAMES = [
    "Ruuvi 2559",
    "Ruuvi BAAD",
    "Ruuvi B69D",
    "Ruuvi 862F",
]

# GATEWAY_POS = (1.5, 0.5)
GATEWAY_POS = (0.5, 2.0)

RUUVI_POSITIONS = {
    # "Ruuvi 2559": (0.0, 0.0),
    # "Ruuvi BAAD": (3.0, 0.0),
    # "Ruuvi B69D": (0.0, 8.0),
    # "Ruuvi 862F": (3.0, 8.0),
    "Ruuvi 2559": (0.0, 0.0),
    "Ruuvi BAAD": (2.5, 0.0),
    "Ruuvi B69D": (0.0, 2.5),
    "Ruuvi 862F": (2.5, 2.5),
}

seeable_info = None
target_ble_list = []


# ---------- Kalman Filter ----------
def apply_kalman_filter(id, rssi, q=1.0, r=4.0):
    if id not in kalman_filters:
        kalman_filters[id] = {"x": rssi, "p": 1.0}
    state = kalman_filters[id]
    k = state["p"] / (state["p"] + r)
    state["x"] = state["x"] + k * (rssi - state["x"])
    state["p"] = (1 - k) * state["p"] + q
    return state["x"]


def send_to_server(data):
    try:
        requests.post("http://localhost:3000/ble-data", json=data)
        print("[ส่งข้อมูล]", data)
    except Exception as e:
        print("ส่งข้อมูลไม่สำเร็จ:", e)


async def monitor_connected_device(device: BLEDevice):
    mac = device.address
    print(f"พยายามเชื่อมต่อกับอุปกรณ์: {device.name}")
    try:
        async with BleakClient(device) as client:
            if await client.is_connected():
                print(f"เชื่อมต่อสำเร็จ: {device.name}")
                while True:
                    if should_send_data and current_uuid:
                        name = (device.name or "").strip()
                        if name in target_ble_names:
                            try:
                                raw_rssi = await client.get_rssi()
                            except:
                                raw_rssi = device.rssi

                            rssi = apply_kalman_filter(mac, raw_rssi)

                            if name in TARGET_BLE_NAMES:
                                if mac not in rssi_history:
                                    rssi_history[mac] = {
                                        "name": name,
                                        "history": deque(maxlen=50),
                                    }
                                rssi_history[mac]["history"].append(rssi)

                            data = {
                                "uuid": current_uuid,
                                "mac": mac,
                                "name": name,
                                "rssi": rssi,
                            }
                            send_to_server(data)
                    await asyncio.sleep(2)
    except Exception as e:
        print(f"ไม่สามารถเชื่อมต่อกับ {device.name}: {e}")
    finally:
        if mac in active_monitors:
            del active_monitors[mac]
            print(f"หยุด monitor อุปกรณ์: {device.name}")


def detection_callback(device: BLEDevice, adv_data: AdvertisementData):
    global seeable_info, target_ble_list

    name = (device.name or adv_data.local_name or "").strip()
    mac = device.address
    raw_rssi = device.rssi

    id = mac  # Default ID
    rssi = apply_kalman_filter(mac, raw_rssi)

    # เช็ค seeable จาก manufacturerData
    for mfg_id, mfg_data in adv_data.manufacturer_data.items():
        try:
            mfg_string = mfg_data.decode("utf-8")
            if mfg_string.startswith(SEEABLE_PREFIX):
                uuid = mfg_string.replace(SEEABLE_PREFIX, "")
                id = uuid
                rssi = apply_kalman_filter(uuid, raw_rssi)
                if uuid not in rssi_history:
                    rssi_history[uuid] = {
                        "name": "seeable",
                        "history": deque(maxlen=20),
                    }
                rssi_history[uuid]["history"].append(rssi)

                seeable_info = {
                    "name": "seeable",
                    "uuid": uuid,
                    "mac": mac,
                    "rssi": rssi,
                }
                break
        except Exception:
            continue

    # สำหรับ Ruuvi
    if name in TARGET_BLE_NAMES:
        if mac not in rssi_history:
            rssi_history[mac] = {"name": name, "history": deque(maxlen=20)}
        rssi_history[mac]["history"].append(rssi)

        existing = next((d for d in target_ble_list if d["mac"] == mac), None)
        if not existing and name in RUUVI_POSITIONS:
            target_ble_list.append({"name": name, "mac": mac, "rssi": rssi})
            print(f"[พบอุปกรณ์เป้าหมาย] {name} {mac} RSSI: {rssi}")

        if mac not in active_monitors:
            print(f"[กำลังเชื่อมต่อกับอุปกรณ์ Ruuvi] {name} ({mac})")
            task = asyncio.create_task(monitor_connected_device(device))
            active_monitors[mac] = task


async def seeable_monitor_loop():
    while True:
        if should_send_data and current_uuid and seeable_info:
            ruuvi_data = []
            for d in target_ble_list:
                mac = d["mac"]
                if (
                    d["name"] in target_ble_names
                    and d["name"] in RUUVI_POSITIONS
                    and mac in rssi_history
                    and rssi_history[mac]["history"]
                ):
                    avg_rssi = sum(rssi_history[mac]["history"]) / len(
                        rssi_history[mac]["history"]
                    )
                    ruuvi_data.append(
                        {
                            "name": d["name"],
                            "rssi": avg_rssi,
                            "position": RUUVI_POSITIONS[d["name"]],
                        }
                    )

            position = None
            seeable_id = seeable_info.get("uuid", seeable_info["mac"])
            if (
                len(ruuvi_data) >= 2
                and seeable_id in rssi_history
                and rssi_history[seeable_id]["history"]
            ):
                try:
                    seeable_avg = sum(rssi_history[seeable_id]["history"]) / len(
                        rssi_history[seeable_id]["history"]
                    )
                    pos = estimate_seeable_position(
                        GATEWAY_POS, ruuvi_data, seeable_avg
                    )
                    position = {"x": round(pos[0], 2), "y": round(pos[1], 2)}
                except Exception as e:
                    print("[ตำแหน่งคำนวณไม่สำเร็จ]", e)

            data = {
                "uuid": current_uuid,
                "mac": seeable_info["mac"],
                "name": seeable_info["name"],
                "rssi": seeable_info["rssi"],
            }

            if position:
                data["position"] = position

            send_to_server(data)
        await asyncio.sleep(2)


async def print_ble_status_loop():
    while True:
        print("\n--- BLE STATUS ---")

        if seeable_info:
            print(f"🔵 Seeable: {seeable_info['mac']} | RSSI: {seeable_info['rssi']}")
        else:
            print("🔵 Seeable: ยังไม่พบ")

        if target_ble_list:
            for dev in target_ble_list:
                print(f"🟢 {dev['name']}: {dev['mac']} | RSSI: {dev['rssi']}")
        else:
            print("🟢 ยังไม่พบอุปกรณ์ Ruuvi ใด ๆ")

        print("\n📶 ประวัติ RSSI ล่าสุด:")
        for key, info in rssi_history.items():
            rssi_list = list(info["history"])
            avg_rssi = round(sum(rssi_list) / len(rssi_list), 2) if rssi_list else None
            print(f"  - {key} ({info['name']}) | avg: {avg_rssi}")

        await asyncio.sleep(2)


def rssi_to_distance(rssi, tx_power=-59, n=3.0):
    return round(10 ** ((tx_power - rssi) / (10 * n)), 2)


def trilaterate(positions, distances):
    A = 2 * (positions[1:] - positions[0])
    b = (
        distances[0] ** 2
        - distances[1:] ** 2
        + np.sum(positions[1:] ** 2, axis=1)
        - np.sum(positions[0] ** 2)
    )
    pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return pos


def estimate_seeable_position(
    gateway_pos, ruuvi_data, seeable_rssi, tx_power=-65, n=3.3
):
    anchor_positions = [gateway_pos]
    anchor_distances = [rssi_to_distance(seeable_rssi, tx_power, n)]

    for ruuvi in ruuvi_data:
        anchor_positions.append(ruuvi["position"])
        d = rssi_to_distance(ruuvi["rssi"], tx_power, n)
        anchor_distances.append(d)

    anchor_positions = np.array(anchor_positions)
    anchor_distances = np.array(anchor_distances)

    return trilaterate(anchor_positions, anchor_distances)


async def ble_loop():
    print("เริ่มสแกนหา RuuviTag BLE...")
    scanner = BleakScanner()
    scanner.register_detection_callback(detection_callback)
    await scanner.start()

    asyncio.create_task(seeable_monitor_loop())
    asyncio.create_task(print_ble_status_loop())

    while True:
        await asyncio.sleep(5)


def ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ble_loop())


@app.route("/control", methods=["POST"])
def control():
    global should_send_data, current_uuid, target_ble_names
    data = request.get_json()
    flag = data.get("flag")
    uuid = data.get("uuid")
    ble_names = data.get("ble_names")

    if flag == "start":
        if not uuid or not ble_names:
            return {"error": "ต้องส่ง uuid และ ble_names"}, 400
        print(f"เริ่มส่งข้อมูล UUID: {uuid}")
        current_uuid = uuid
        target_ble_names = ble_names
        should_send_data = True
        return {"status": "started"}

    elif flag == "stop":
        print("หยุดส่งข้อมูล")
        should_send_data = False
        current_uuid = None
        target_ble_names = []
        return {"status": "stopped"}

    return {"error": "invalid flag"}, 400


if __name__ == "__main__":
    threading.Thread(target=ble_thread, daemon=True).start()
    app.run(port=5001)
