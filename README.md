# Gyra
Repo to use yolo detectors at 1k+ RPM of the source video.
ffmpeg -i cars_144p.mp4 -ss 00:00:00 -to 00:02:00 -c copy cars__144p.mp4
ffmpeg -i cars_240.mp4 -ss 00:00:00 -to 00:02:00 -c copy cars__240p.mp4
ffmpeg -i cars_360.mp4 -ss 00:00:00 -to 00:02:00 -c copy cars__360p.mp4
ffmpeg -i cars_480.mp4 -ss 00:00:00 -to 00:02:00 -c copy cars__480p.mp4


python spin_video.py "data/cars__480p.mp4" --rpm 3000 --out "data/out/480_3k.mp4" --blur 0
python detect_vehicles.py "data/out/144.mp4"

python spin_video_realistic.py "D:\github_repos_windows\Gyra\src\data\cars__480p.mp4" --rpm 1200 --out "data\realistic_480p.mp4" --accel-duration 4.0 --oscillation-factor 0.12 --oscillation-period 3.0 --blur 5 --no-overlay

python telemetry_detect_vehicles.py "data\realistic_480p.mp4" --rpm 1200 --output "data\realistic_output.mp4"
python vidstab_detect_vehicles.py "data\realistic_480p.mp4" --rpm 1200 --output "data\realistic_output.mp4"

