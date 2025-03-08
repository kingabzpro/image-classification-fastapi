# Model info endpoints
curl -X GET 'http://127.0.0.1:5454/model-info'

curl -X GET 'http://127.0.0.1:5454/model-info' \
  -H 'accept: application/json' \
  -H 'X-API-Key: asfasfwefasasfasdf'

curl -X GET 'http://127.0.0.1:5454/model-info' \
  -H 'accept: application/json' \
  -H 'X-API-Key: q3hewf#onio12$r032'

# Predict endpoints
curl -X POST 'http://127.0.0.1:5454/predict' \
  -H 'accept: application/json' \
  -H 'X-API-Key: asfasfwefasasfasdf' \
  -F 'file=@./data/sample/cat.png'

curl -X POST 'http://127.0.0.1:5454/predict' \
  -H 'accept: application/json' \
  -H 'X-API-Key: q3hewf#onio12$r032' \
  -F 'file=@./data/sample/cat.png'