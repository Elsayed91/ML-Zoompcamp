import bentoml
from io import BytesIO
import numpy as np
from bentoml.io import Image
from bentoml.io import JSON

runner = bentoml.keras.get("kitchenware-classifier:latest").to_runner()
svc = bentoml.Service("kitchenware-classifier-service", runners=[runner])

@svc.api(input=Image(), output=JSON())

async def predict(img):

    from tensorflow.keras.applications.xception import preprocess_input

    img = img.resize((299, 299))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = await runner.async_run(arr)
    classes = ['cup', 'fork', 'glass', 'knife', 'plate', 'spoon']
    prediction = classes[preds.argmax(axis=1)[0]]
    return {
        'prediction': prediction
    }
