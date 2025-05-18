import asyncio

import numpy as np
from bleak import AdvertisementData, BleakScanner, BLEDevice

SEEABLE_SIGNATURE = b"seeable"
TARGET_BLE_NAMES = [
    "Ruuvi 2559",
    "Ruuvi BAAD",
    "Ruuvi B69D",
    "Ruuvi 862F",
    # "Ruuvi 30E9",
]

GATEWAY_POS = (0.5, 2.0)  # (0.5, 2.0) (1.5, 0.5)

RUUVI_POSITIONS = {
    # "Ruuvi 2559": (0.0, 0.0),
    # "Ruuvi BAAD": (3.0, 0.0),
    # "Ruuvi B69D": (0.0, 8.0),
    # "Ruuvi 862F": (3.0, 8.0),
    "Ruuvi 2559": (0.0, 0.0),
    "Ruuvi BAAD": (2.5, 0.0),
    "Ruuvi B69D": (0.0, 2.5),
    "Ruuvi 862F": (2.5, 2.5),
    # "Ruuvi 30E9": (2.0, 0.0),
}

seeable_info = None
target_ble_list = []


# ---------- calculate distance ----------
def rssi_to_distance(rssi, tx_power=-59, n=3.0):
    return round(10 ** ((tx_power - rssi) / (10 * n)), 2)


# ---------- Trilateration ----------
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
    gateway_pos, ruuvi_data, seeable_rssi, tx_power=-59, n=2.0
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


# ---------- Callback ----------
def detection_callback(device: BLEDevice, adv_data: AdvertisementData):
    global seeable_info, target_ble_list

    name = (device.name or adv_data.local_name or "").strip()
    mac = device.address
    rssi = device.rssi

    # find seeable
    for mfg_id, mfg_data in adv_data.manufacturer_data.items():
        if SEEABLE_SIGNATURE in mfg_data and seeable_info is None:
            seeable_info = {
                "name": "seeable",
                "mac": mac,
                "rssi": rssi,
                "distance": rssi_to_distance(rssi),
            }
            print(
                f"[พบ seeable จาก manufacturer data] {mac} RSSI: {rssi} → ระยะ ≈ {seeable_info['distance']} ม."
            )
            break

    # filter Ruuvi
    if name in TARGET_BLE_NAMES:
        existing = next((d for d in target_ble_list if d["mac"] == mac), None)
        if not existing:
            if name in RUUVI_POSITIONS:
                info = {
                    "name": name,
                    "mac": mac,
                    "rssi": rssi,
                    "distance": rssi_to_distance(rssi),
                    "position": RUUVI_POSITIONS[name],
                }
                target_ble_list.append(info)
                print(
                    f"[พบอุปกรณ์เป้าหมาย] {name} {mac} RSSI: {rssi} → ระยะ ≈ {info['distance']} ม."
                )


# ---------- Main Scan ----------
async def scan_ble():
    scanner = BleakScanner()
    scanner.register_detection_callback(detection_callback)

    print("กำลังสแกน BLE (10 วินาที)...")
    await scanner.start()
    await asyncio.sleep(10)
    await scanner.stop()

    if not seeable_info:
        print("ไม่พบอุปกรณ์ seeable จาก manufacturer data")
        return

    if not target_ble_list:
        print("ไม่พบ BLE เป้าหมายใด ๆ ที่ตรงชื่อ")
        return

    print("\nสรุปผลการคำนวณ:")
    print(
        f"- seeable ({seeable_info['mac']}) RSSI: {seeable_info['rssi']} → ระยะ ≈ {seeable_info['distance']} ม."
    )

    for target in target_ble_list:
        diff = abs(seeable_info["distance"] - target["distance"])
        print(
            f"- {target['name']} ({target['mac']}) RSSI: {target['rssi']} → ระยะ ≈ {target['distance']} ม."
        )
        print(f"  → ห่างจาก seeable ≈ {diff:.2f} เมตร")

    # ---------- calculate seeable position ----------
    if len(target_ble_list) >= 2:  # target must be founded more than 2
        estimated_pos = estimate_seeable_position(
            GATEWAY_POS, target_ble_list, seeable_info["rssi"]
        )
        print(
            f"\nตำแหน่งโดยประมาณของ seeable: x={estimated_pos[0]:.2f}, y={estimated_pos[1]:.2f}"
        )
    else:
        print("\nไม่สามารถประมาณตำแหน่ง seeable ได้ (ต้องมี Ruuvi อย่างน้อย 2 จุด)")


# ---------- Run ----------
if __name__ == "__main__":
    asyncio.run(scan_ble())

# ------------------------------------------------------------------------------------------------------------------------------------------------------

# import asyncio
# import threading

# import numpy as np
# import requests
# from bleak import AdvertisementData, BleakClient, BleakScanner, BLEDevice
# from flask import Flask, request

# # ---------- Flask Setup ----------
# app = Flask(__name__)
# should_send_data = False
# current_uuid = None
# target_ble_names = []
# active_monitors = {}
# RUUVI_MANUFACTURER_ID = 0x0499
# SEEABLE_SIGNATURE = b"seeable"

# # ---------- BLE Positioning Setup ----------
# GATEWAY_POS = (0.5, 2.0)
# RUUVI_POSITIONS = {
#     "Ruuvi 2559": (0.0, 0.0),
#     "Ruuvi BAAD": (2.5, 0.0),
#     "Ruuvi B69D": (0.0, 2.5),
#     "Ruuvi 862F": (2.5, 2.5),
# }

# seeable_info = None
# target_ble_list = []


# # ---------- Utils ----------
# def rssi_to_distance(rssi, tx_power=-59, n=2.0):
#     return round(10 ** ((tx_power - rssi) / (10 * n)), 2)


# def trilaterate(positions, distances):
#     A = 2 * (positions[1:] - positions[0])
#     b = (
#         distances[0] ** 2
#         - distances[1:] ** 2
#         + np.sum(positions[1:] ** 2, axis=1)
#         - np.sum(positions[0] ** 2)
#     )
#     pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
#     return pos


# def estimate_seeable_position(gateway_pos, ruuvi_data, seeable_rssi):
#     anchor_positions = [gateway_pos]
#     anchor_distances = [rssi_to_distance(seeable_rssi)]

#     for ruuvi in ruuvi_data:
#         anchor_positions.append(ruuvi["position"])
#         d = rssi_to_distance(ruuvi["rssi"])
#         anchor_distances.append(d)

#     anchor_positions = np.array(anchor_positions)
#     anchor_distances = np.array(anchor_distances)
#     return trilaterate(anchor_positions, anchor_distances)


# # ---------- Server Communication ----------
# def send_to_server(data):
#     try:
#         requests.post("http://localhost:3000/ble-data", json=data)
#         print("[ส่งข้อมูล]", data)
#     except Exception as e:
#         print("ส่งข้อมูลไม่สำเร็จ:", e)


# # ---------- BLE Monitor Logic ----------
# async def monitor_connected_device(device: BLEDevice):
#     global seeable_info, target_ble_list
#     mac = device.address
#     print(f"พยายามเชื่อมต่อกับอุปกรณ์: {mac}")
#     try:
#         async with BleakClient(device) as client:
#             if await client.is_connected():
#                 print(f"เชื่อมต่อสำเร็จ: {mac}")
#                 while True:
#                     if should_send_data and current_uuid:
#                         name = (device.name or "").strip()
#                         try:
#                             rssi = await client.get_rssi()
#                         except:
#                             rssi = device.rssi

#                         if name == "seeable":
#                             seeable_info = {
#                                 "name": name,
#                                 "mac": mac,
#                                 "rssi": rssi,
#                                 "distance": rssi_to_distance(rssi),
#                             }
#                         elif name in RUUVI_POSITIONS:
#                             found = next(
#                                 (x for x in target_ble_list if x["mac"] == mac), None
#                             )
#                             if not found:
#                                 target_ble_list.append(
#                                     {
#                                         "name": name,
#                                         "mac": mac,
#                                         "rssi": rssi,
#                                         "distance": rssi_to_distance(rssi),
#                                         "position": RUUVI_POSITIONS[name],
#                                     }
#                                 )

#                         # ส่งข้อมูลทั้งหมด
#                         data = {
#                             "uuid": current_uuid,
#                             "mac": mac,
#                             "name": name,
#                             "rssi": rssi,
#                         }
#                         send_to_server(data)

#                         # ส่งตำแหน่ง seeable ถ้าคำนวณได้
#                         if seeable_info and len(target_ble_list) >= 2:
#                             est_pos = estimate_seeable_position(
#                                 GATEWAY_POS, target_ble_list, seeable_info["rssi"]
#                             )
#                             position_data = {
#                                 "uuid": current_uuid,
#                                 "mac": seeable_info["mac"],
#                                 "name": "seeable_position",
#                                 "x": round(est_pos[0], 2),
#                                 "y": round(est_pos[1], 2),
#                             }
#                             send_to_server(position_data)

#                     await asyncio.sleep(2)
#     except Exception as e:
#         print(f"ไม่สามารถเชื่อมต่อกับ {device.name}: {e}")
#     finally:
#         if mac in active_monitors:
#             del active_monitors[mac]
#             print(f"หยุด monitor อุปกรณ์: {device.name}")


# # ---------- BLE Background Loop ----------
# async def ble_loop():
#     print("เริ่มสแกนหาอุปกรณ์ BLE...")
#     scanner = BleakScanner()
#     while True:
#         if should_send_data:
#             devices = await scanner.discover(timeout=5.0)
#             for device in devices:
#                 name = (device.name or "").strip()
#                 if device.address in active_monitors:
#                     continue
#                 if name == "seeable" or name in target_ble_names:
#                     task = asyncio.create_task(monitor_connected_device(device))
#                     active_monitors[device.address] = task
#         await asyncio.sleep(5)


# # ---------- Start Background Thread ----------
# def ble_thread():
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(ble_loop())


# # ---------- Flask Endpoint ----------
# @app.route("/control", methods=["POST"])
# def control():
#     global should_send_data, current_uuid, target_ble_names, seeable_info, target_ble_list

#     data = request.get_json()
#     flag = data.get("flag")
#     uuid = data.get("uuid")
#     ble_names = data.get("ble_names")

#     if flag == "start":
#         if not uuid or not ble_names:
#             return {"error": "ต้องส่ง uuid และ ble_names"}, 400
#         print(f"เริ่มส่งข้อมูล UUID: {uuid}")
#         current_uuid = uuid
#         target_ble_names = ble_names
#         seeable_info = None
#         target_ble_list = []
#         should_send_data = True
#         return {"status": "started"}

#     elif flag == "stop":
#         print("หยุดส่งข้อมูล")
#         should_send_data = False
#         current_uuid = None
#         target_ble_names = []
#         seeable_info = None
#         target_ble_list = []
#         return {"status": "stopped"}

#     return {"error": "invalid flag"}, 400


# # ---------- Start Server ----------
# if __name__ == "__main__":
#     threading.Thread(target=ble_thread, daemon=True).start()
#     app.run(port=5001)

# ------------------------------------------------------------------------------------------------------------------------------------------------------
