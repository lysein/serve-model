from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np

# Load the model
model = xgb.Booster()
model.load_model("./xgboost-best.model")

app = FastAPI()

class InputData(BaseModel):
    bar: float
    baz: float
    xgt: float
    qgg: float
    lux: float
    wsg: float
    yyz: float
    drt: float
    gox: float
    foo: float
    boz: float
    fyt: float
    lgh: float
    hrt: float
    juu: float
    day: float
    month: float
    year: float
    day_of_week: float
    is_weekend: int

@app.post("/predict")
def predict(data: InputData):
    try:
        input_data = np.array([[
            data.bar, data.baz, data.xgt, data.qgg, data.lux, data.wsg,
            data.yyz, data.drt, data.gox, data.foo, data.boz, data.fyt,
            data.lgh, data.hrt, data.juu, data.day, data.month, data.year,
            data.day_of_week, data.is_weekend
        ]])
        dmatrix = xgb.DMatrix(input_data)
        prediction = model.predict(dmatrix)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
