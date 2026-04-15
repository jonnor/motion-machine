

## Practical test

- Do some data recording of known activities.

## Watch UI

- Add some graphics on screen. Using micropython-touch
- handle button press wakeup. Enable POWERON IRQ in AXP2101, IQR handler on pin 21
- Ensure can see record status on watch.
- Add enable/disable recording to watch UI?
- Add watchdog to ensure no hangups?
- Allow to add labels via screen

## Power Management

- Use lightsleep when possible.
Ideally transparently with asyncio, using asyncio_alt.
https://github.com/peterhinch/micropython-async/blob/master/v3/asyncio_alt/core.py
- Only turn on WiFi when in a WiFi mode
- Support easy configuration of WiFi

## Connectivity

- Support sending predictions out via BLE

## Nice to have:

- Event log. That can be accessed via API/webui.
Internal actions, external.
- Discovery device of using mDNS
- Easier way to configure WiFi credentials
- Ability to act as WiFi station
