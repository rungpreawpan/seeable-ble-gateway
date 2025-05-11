import asyncio
import threading

import requests
from bleak import AdvertisementData, BleakClient, BleakScanner, BLEDevice
from flask import Flask, request

app = Flask(__name__)
should_send_data = False
current_uuid = None
target_ble_names = []
active_monitors = {}
RUUVI_MANUFACTURER_ID = 0x0499


def send_to_server(data):
    try:
        requests.post("http://localhost:3000/ble-data", json=data)
        print("[ส่งข้อมูล]", data)
    except Exception as e:
        print("ส่งข้อมูลไม่สำเร็จ:", e)


async def monitor_connected_device(device: BLEDevice):
    mac = device.address
    print(f"พยายามเชื่อมต่อกับอุปกรณ์: {mac}")
    try:
        async with BleakClient(device) as client:
            if await client.is_connected():
                print(f"เชื่อมต่อสำเร็จ: {mac}")
                while True:
                    if should_send_data and current_uuid:
                        name = (device.name or "").strip()
                        if name in target_ble_names:
                            try:
                                rssi = await client.get_rssi()
                            except:
                                rssi = device.rssi
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
    mac = device.address
    if mac in active_monitors:
        return

    if RUUVI_MANUFACTURER_ID in adv_data.manufacturer_data:
        print(f"พบ RuuviTag: {mac} | RSSI: {device.rssi}")
        task = asyncio.create_task(monitor_connected_device(device))
        active_monitors[mac] = task


async def ble_loop():
    print("เริ่มสแกนหา RuuviTag BLE...")
    scanner = BleakScanner()
    scanner.register_detection_callback(detection_callback)
    await scanner.start()
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
