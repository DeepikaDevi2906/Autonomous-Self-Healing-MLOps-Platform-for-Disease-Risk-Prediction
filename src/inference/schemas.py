"""
Request / response models. Field ranges reject obviously invalid input with a 422
(the old schemas accepted anything, e.g. a negative glucose).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, create_model


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiabetesInput(_Input):
    Pregnancies: int = Field(ge=0, le=25)
    Glucose: float = Field(ge=0, le=400, description="0 is treated as 'not measured'")
    BloodPressure: float = Field(ge=0, le=250, description="0 is treated as 'not measured'")
    SkinThickness: float = Field(ge=0, le=120, description="0 is treated as 'not measured'")
    Insulin: float = Field(ge=0, le=1200, description="0 is treated as 'not measured'")
    BMI: float = Field(ge=0, le=90, description="0 is treated as 'not measured'")
    DiabetesPedigreeFunction: float = Field(ge=0, le=5)
    Age: int = Field(ge=1, le=120)

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
        "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}})


class HeartDiseaseInput(_Input):
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(ge=0, le=3, description="chest pain type")
    trestbps: float = Field(ge=50, le=250, description="resting blood pressure")
    chol: float = Field(ge=50, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=50, le=250, description="max heart rate")
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4, description="4 = unknown")
    thal: int = Field(ge=0, le=3, description="0 = unknown")

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
        "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}})


_BC_FIELDS = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
    "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry",
    "mean_fractal_dimension", "radius_error", "texture_error", "perimeter_error",
    "area_error", "smoothness_error", "compactness_error", "concavity_error",
    "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension",
]

_BC_EXAMPLE = dict(zip(_BC_FIELDS, [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
]))


BreastCancerInput = create_model(
    "BreastCancerInput",
    __base__=_Input,
    **{name: (float, Field(ge=0)) for name in _BC_FIELDS},
)
BreastCancerInput.model_config["json_schema_extra"] = {"example": _BC_EXAMPLE}
BreastCancerInput.model_rebuild(force=True)


INPUT_SCHEMAS = {
    "diabetes": DiabetesInput,
    "heart_disease": HeartDiseaseInput,
    "breast_cancer": BreastCancerInput,
}


class PredictionResponse(BaseModel):
    disease: str
    prediction: int = Field(description="1 = disease present, 0 = not present")
    prediction_label: str
    probability: float = Field(description="predicted probability that the disease is present")
    model_version: str
    model_algorithm: str


class RollbackResponse(BaseModel):
    disease: str
    restored_version: str | None
    message: str


class BatchRequest(BaseModel):
    rows: int | None = Field(default=None, ge=30, le=5000, description="default: whole future pool")
    drift: float = Field(default=0.0, ge=0, le=5, description="shift in standard deviations")
    label_noise: float = Field(default=0.0, ge=0, le=0.5, description="random label flips")
    concept_shift: float = Field(default=0.0, ge=0, le=1, description="learnable label change")


class BatchUpload(BaseModel):
    filename: str = Field(default="upload.csv", max_length=200)
    csv: str = Field(max_length=5_000_000, description="the CSV file's text")
