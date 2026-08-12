from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parent


def test_weekly_workflow_exists_and_is_strict():
    workflow = (REPO_ROOT / ".github" / "workflows" / "train_models.yml").read_text()
    assert 'cron: "0 0 * * 0"' in workflow
    assert "python -m services.model_selector --strict" in workflow
    assert "python scripts/export_forecast_artifacts.py" in workflow
    assert "python scripts/validate_exports.py" in workflow
    assert "contents: write" in workflow
    assert "continue-on-error" not in workflow


def test_training_and_inference_dependencies_are_aligned():
    training = (BACKEND / "requirements-pipeline.txt").read_text()
    inference = (BACKEND / "requirements-inference.txt").read_text()

    packages = [
        "pandas==3.0.5",
        "numpy==1.26.4",
        "scikit-learn==1.9.0",
        "scipy==1.17.1",
        "statsmodels==0.14.6",
        "torch==2.13.0",
        "joblib==1.5.3",
    ]
    for package in packages:
        assert package in training
        assert package in inference
