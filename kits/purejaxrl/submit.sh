tar -czf submission.tar.gz $(find . -maxdepth 1 -type f) models/latest_model.pkl ./distrax/ ./lux/ ./luxai_s3_local/
mv submission.tar.gz logs