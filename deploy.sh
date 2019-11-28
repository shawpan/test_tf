# Start tensorflow serving
docker run -p 8501:8501 \
  --mount type=bind,source="$(pwd)"/export/,target=/models/my_model \
  -e MODEL_NAME=my_model -t tensorflow/serving
