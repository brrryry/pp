####################################
# General configuration file for the network
####################################

# Path to the directory where the data is stored
maps_path = "./data/maps"

# Path to the directory where the logs are stored
log_path = "./data/logs"

# API link to the osu! API
mirrors = [
    ("Nerinyan", f"https://api.nerinyan.moe/d/"),
    ("OsuDirect", f"https://osu.direct/api/d/"),
    ("Chimu", f"https://api.chimu.moe/v1/download/"),
    ("Sayobot", f"https://txy1.sayobot.cn/beatmaps/download/full/")
]

# osupy api credentials
osu_api_client_id = 21492
osu_api_client_secret = "xy9N7M6VKU3Bx2cboaoqddRTcHmTHxiEZE49re1d"
osu_api_redirect_uri = "http://localhost:4000"

# Watcher configuration
# Set to True if you want this server instance to also monitor local folders for replays
run_watcher = False