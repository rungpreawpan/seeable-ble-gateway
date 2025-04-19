from flask import Flask, request
import asyncio
import threading
import requests
from bleak import BleakScanner, BleakClient

app = Flask(__name__)
should_navigate = False
current_uuid = None
found_ble_macs = set()

def send_to_server(data):
    try:
        requests.post('http://localhost:3000/ble-data', json=data)
    except Exception as e:
        print("ส่งข้อมูลไปยัง Server ไม่สำเร็จ:", e)

# ส่งข้อมูล BLE ไปเรื่อยๆ ทุก 2 วินาที ตราบใดที่ should_navigate ยังเป็น True
async def connect_and_stream(device, uuid):
    try:
        async with BleakClient(device) as client:
            print(f"เชื่อมต่อกับ {device.address}")
            while should_navigate:
                data = {
                    "uuid": uuid,
                    "mac": device.address,
                    "name": device.name,
                    "rssi": device.rssi
                }
                send_to_server(data)
                await asyncio.sleep(2)
    except Exception as e:
        print(f"⚠️ ไม่สามารถเชื่อมต่อ {device.address}: {e}")

# วนสแกนตลอด (แม้เจอครบแล้วก็ไม่หยุด)
async def ble_loop():
    global found_ble_macs
    found_ble_macs = set()

    while should_navigate:
        print(f"สแกน BLE... (เจอแล้ว {len(found_ble_macs)} ตัว)")
        devices = await BleakScanner.discover(timeout=5)

        for device in devices:
            if "Ruuvi" in (device.name or "") and device.address not in found_ble_macs:
                found_ble_macs.add(device.address)
                asyncio.create_task(connect_and_stream(device, current_uuid))

        await asyncio.sleep(3)  # เว้นช่วงสแกนรอบใหม่

    print("หยุดสแกนแล้ว")

# ใช้ event loop แยก thread (รองรับ macOS)
def ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ble_loop())

@app.route('/control', methods=['POST'])
def control():
    global should_navigate, current_uuid
    data = request.get_json()
    flag = data.get("flag")
    uuid = data.get("uuid")

    if flag == "start":
        if not uuid:
            return {"error": "ต้องมี uuid"}, 400

        print(f"▶️ เริ่มนำทาง UUID: {uuid}")
        current_uuid = uuid
        should_navigate = True
        threading.Thread(target=ble_thread).start()
        return {"status": "started"}

    elif flag == "stop":
        print("หยุดนำทาง")
        should_navigate = False
        current_uuid = None
        return {"status": "stopped"}

    return {"error": "invalid flag"}, 400

if __name__ == '__main__':
    app.run(port=5001)
