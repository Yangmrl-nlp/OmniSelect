cd /mnt/data2/yangmrl/project/video2text/test_data/worldsense/videos

for file in *.mp4; do
    ffmpeg -i "$file" -vn -ac 1 -ar 16000 -y "/mnt/data2/yangmrl/project/video2text/test_data/worldsense/audios/${file%.*}.wav"
done