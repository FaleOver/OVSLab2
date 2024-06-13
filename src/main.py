# -*- coding: utf-8 -*-
import os
from model_utils import load_model, make_inference
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi_utils import Oauth2ClientCredentials
from pydantic import BaseModel
from keycloak.uma_permissions import AuthStatus
from keycloak_utils import get_keycloak_data
import requests


class Instance(BaseModel):
    Pclass: int
    Sex: str
    Age: int
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


class Client(BaseModel):
    client_id: str
    client_secret: str


class CombinedData(BaseModel):
    instance: Instance
    client: Client


app = FastAPI()
keycloak_openid, token_endpoint = get_keycloak_data()
oauth2_scheme = Oauth2ClientCredentials(tokenUrl=token_endpoint)

model_path: str = os.getenv("MODEL_PATH")
if model_path is None:
    raise ValueError("The environment variable $MODEL_PATH is empty!")


async def get_token_status(token: str) -> AuthStatus:
    return keycloak_openid.has_uma_access(
        token, "infer_endpoint#doInfer")


async def check_token(token: str = Depends(oauth2_scheme)) -> None:
    print("---")
    print(token)
    print("---")
    auth_status = await get_token_status(token)
    is_logged = auth_status.is_logged_in
    is_authorized = auth_status.is_authorized

    print("---")
    print(auth_status)
    print(token_endpoint)
    print("---")

    if not is_logged:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    elif not is_authorized:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    print(token_endpoint)
    return {"status": "ok"}


@app.post("/predictions")
async def predictions_api(data: CombinedData) -> dict[str, float]:
    """Возвращает результат предсказания модели для in_data."""
    instance = data.instance
    client = data.client

    base_url = "https://keycloak:8443"
    realm_path = "/realms/inference"
    token_path = "/protocol/openid-connect/token"
    url = base_url + realm_path + token_path
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'client_id': client.client_id,
        'client_secret': client.client_secret,
        'grant_type': 'client_credentials'
    }
    response = requests.post(url, headers=headers, data=data, verify=False)

    if response.status_code == 200:
        access_token = response.json().get('access_token')
        if access_token:
            await check_token(access_token)
            return make_inference(load_model(model_path), instance.dict())
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return make_inference(load_model(model_path), instance.dict())
