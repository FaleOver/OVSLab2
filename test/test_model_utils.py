# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from model_utils import make_inference, load_model
from sklearn.pipeline import Pipeline
from pickle import dumps


@pytest.fixture
def create_data() -> dict[str, int | float | str]:
    return {"Pclass": 1, "Sex": "male", "Age": 18, "SibSp": 2,
            "Parch": 1, "Fare": 35.26, "Embarked": "B"}


def test_make_inference(monkeypatch, create_data):
    def mock_get_predictions(_, data: pd.DataFrame) -> list[list[float]]:
        assert create_data == {
            key: value[0] for key, value in data.to_dict("list").items()
        }
        return [1]

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict", mock_get_predictions)

    result = make_inference(in_model, create_data)
    assert result == {"survived": 1}


def test_make_real_inference(monkeypatch, create_data):
    result = make_inference(load_model("./models/pipeline.pkl"), create_data)
    assert result == {"survived": 1}


@pytest.fixture()
def filepath_and_data(tmpdir):
    p = tmpdir.mkdir("datadir").join("fakedmodel.pkl")
    example: str = "Test message!"
    p.write_binary(dumps(example))
    return str(p), example


def test_load_model(filepath_and_data):
    assert filepath_and_data[1] == load_model(filepath_and_data[0])
