import asyncio
import threading
import requests
from bleak import BleakClient, BleakScanner
from flask import Flask, request

app = Flask(__name__)
should_navigate = False
current_uuid = None
target_ble_names = []
found_ble_macs = set()

def send_to_server(data):
    try:
        requests.post("http://localhost:3000/ble-data", json=data)
        print(data)
    except Exception as e:
        print("ส่งข้อมูลไปยัง Server ไม่สำเร็จ:", e)


async def connect_and_stream(device, uuid):
    try:
        async with BleakClient(device) as client:
            print(f"เชื่อมต่อกับ {device.address} ({device.name})")
            while should_navigate:
                data = {
                    "uuid": uuid,
                    "mac": device.address,
                    "name": device.name,
                    "rssi": device.rssi,
                }
                send_to_server(data)
                await asyncio.sleep(2)
    except Exception as e:
        print(f"ไม่สามารถเชื่อมต่อ {device.address} ({device.name}): {e}")


async def ble_loop():
    global found_ble_macs
    found_ble_macs = set()

    while should_navigate:
        devices = await BleakScanner.discover(timeout=5)
        names_found = []

        for device in devices:
            if device.name in target_ble_names and device.address not in found_ble_macs:
                found_ble_macs.add(device.address)
                asyncio.create_task(connect_and_stream(device, current_uuid))

        if names_found:
            print(
                f"กำลังสแกน... เจอแล้ว {len(found_ble_macs)} ตัว: {', '.join(names_found)}"
            )
        else:
            print(f"กำลังสแกน...")

        await asyncio.sleep(3)

    print("หยุดสแกนแล้ว")


def ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ble_loop())


@app.route("/control", methods=["POST"])
def control():
    global should_navigate, current_uuid, target_ble_names
    data = request.get_json()
    flag = data.get("flag")
    uuid = data.get("uuid")
    ble_names = data.get("ble_names")

    if flag == "start":
        if not uuid or not ble_names:
            return {"error": "ต้องส่ง uuid และ ble_names"}, 400

        print(f"เริ่มนำทาง UUID: {uuid} กับ BLE: {ble_names}")
        current_uuid = uuid
        target_ble_names = ble_names
        should_navigate = True
        threading.Thread(target=ble_thread).start()
        return {"status": "started"}

    elif flag == "stop":
        print("หยุดนำทาง")
        should_navigate = False
        current_uuid = None
        target_ble_names = []
        return {"status": "stopped"}

    return {"error": "invalid flag"}, 400


if __name__ == "__main__":
    app.run(port=5001)
