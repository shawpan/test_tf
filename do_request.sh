curl -X POST \
  http://localhost:8501/v1/models/my_model:regress \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -d '{
  "signature_name": "regression",
  "examples": [{
  	"feature1": 1.1,
  	"feature2": 2.1,
  	"feature3": 3.1,
  	"feature4": 1,
  	"feature5": 2,
  	"feature6": 3

  }]
}'
