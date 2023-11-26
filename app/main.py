import uvicorn
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()


class Features(BaseModel):
    Year: int
    Age: int
    Height: int
    Weight: int
    Sprint_40yd: int
    Vertical_Jump: int
    Bench_Press_Reps: int
    Broad_Jump: int
    Agility_3cone: int
    Shuttle: int
    BMI: int
    Player_Type_defense: int
    Player_Type_offense: int
    Player_Type_special_teams: int
    Position_Type_backs_receivers: int
    Position_Type_defensive_back: int
    Position_Type_defensive_lineman: int
    Position_Type_kicking_specialist: int
    Position_Type_line_backer: int
    Position_Type_offensive_lineman: int
    Position_Type_other_special: int
    Position_C: int
    Position_CB: int
    Position_DE: int
    Position_DT: int
    Position_FB: int
    Position_FS: int
    Position_ILB: int
    Position_K: int
    Position_LS: int
    Position_OG: int
    Position_OLB: int
    Position_OT: int
    Position_P: int
    Position_QB: int
    Position_RB: int
    Position_SS: int
    Position_TE: int
    Position_WR: int
#------------------------------------------------

mlflow.set_tracking_uri('https://dagshub.com/ElChicoYetix/PROYECTOFINAL.mlflow')
logged_model = 'runs:/12a04e586ca549feb9ccbf69cf0cba48/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

#------------------------------------------------


@app.get("/api/v0/classify")
async def predict_churn(features: Features):

    # Este comadno convierte los atributos del objeto features en un diccionario
    data = features.__dict__
    df = pd.DataFrame([data['Year'], data['Age'], data['Height'], data['Weight'],
                   data['Sprint_40yd'],
                   data['Vertical_Jump'],
                   data['Bench_Press_Reps'],
                   data['Broad_Jump'],
                   data['Agility_3cone'],
                   data['Shuttle'],
                   data['BMI'],
                   data['Player_Type_defense'],
                   data['Player_Type_offense'],
                   data['Player_Type_special_teams'],
                   data['Position_Type_backs_receivers'],
                   data['Position_Type_defensive_back'],
                   data['Position_Type_defensive_lineman'],
                   data['Position_Type_kicking_specialist'],
                   data['Position_Type_line_backer'],
                   data['Position_Type_offensive_lineman'],
                   data['Position_Type_other_special'],
                   data['Position_C'],
                   data['Position_CB'],
                   data['Position_DE'],
                   data['Position_DT'],
                   data['Position_FB'],
                   data['Position_FS'],
                   data['Position_ILB'],
                   data['Position_K'],
                   data['Position_LS'],
                   data['Position_OG'],
                   data['Position_OLB'],
                   data['Position_OT'],
                   data['Position_P'],
                   data['Position_QB'],
                   data['Position_RB'],
                   data['Position_SS'],
                   data['Position_TE'],
                   data['Position_WR']
                  ]).T


    prediction = loaded_model.predict(df)

    if prediction[0] == 1:
        salida = "Drafted"
    else:
        salida = "Not Drafted"

    return {"Player will be": salida}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=False)
