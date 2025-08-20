# -*- coding: utf-8 -*-

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import uuid

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Passerelle d'Intelligence Artificielle pour l'Imagerie Médicale",
    description="Passerelle pour l'inférence MONAI et la génération de rapports assistés par GPT‑5",
    version="1.0.0"
)

# CORS middleware – In production, restrict the allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class ReportRequest(BaseModel):
    """Requête pour la génération de rapport radiologique."""
    study_id: str
    findings: Dict[str, Any]  # volumes, diamètres, localisations, version du modèle, etc.
    clinical_context: Optional[Dict[str, Any]] = None
    priority: str = "normal"  # normal, urgent

class ReportResponse(BaseModel):
    """Réponse de génération de rapport."""
    report_id: str
    status: str
    structured_report: Optional[Dict[str, Any]] = None
    radiology_narrative: Optional[str] = None
    patient_summary: Optional[str] = None
    generated_at: datetime

class InferenceRequest(BaseModel):
    """Requête d'inférence MONAI."""
    modality: str = "CT"
    model_version: str = "v1.0"
    processing_priority: str = "normal"

# --------------------------------------------------------------------------- #
# In‑memory store (replace with a real DB in production)
# --------------------------------------------------------------------------- #

reports_db: Dict[str, Any] = {}
inference_jobs: Dict[str, Any] = {}

# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/")
async def root():
    """Point d’entrée principal."""
    return {"message": "Passerelle d'Intelligence Artificielle pour l'Imagerie Médicale", "status": "en ligne"}

@app.get("/health")
async def health_check():
    """Check‑up de santé de l’application."""
    return {"status": "sain", "timestamp": datetime.utcnow()}

# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

@app.post("/infer/{study_id}")
async def run_inference(
    study_id: str,
    file: UploadFile = File(...),
    modality: str = Form("CT"),
    model_version: str = Form("v1.0"),
    background_tasks: BackgroundTasks = None
):
    """Traiter l'image médicale et exécuter l'inférence MONAI."""
    try:
        # Validation du type de fichier
        allowed_types = [".nii", ".nii.gz", ".dcm", ".zip"]
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_types and not any(file.filename.lower().endswith(ext) for ext in allowed_types):
            raise HTTPException(
                400,
                f"Type de fichier non pris en charge. Autorisés : {', '.join(allowed_types)}"
            )

        # Sauvegarde temporaire du fichier reçu
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Démarrage de la tâche d'arrière‑plan
        background_tasks.add_task(process_inference, study_id, tmp_path, modality, model_version)

        return {
            "study_id": study_id,
            "status": "en cours de traitement",
            "message": "Job d'inférence démarré",
            "job_id": f"job_{study_id}",
            "estimated_completion": "30 secondes"
        }

    except Exception as e:
        logger.error(f"Erreur d'inférence : {str(e)}")
        raise HTTPException(500, f"Échec de l'inférence : {str(e)}")

async def process_inference(study_id: str, file_path: str, modality: str, model_version: str):
    """Tâche d'arrière‑plan pour le traitement d'inférence."""
    try:
        # Simulation d’un délai de traitement
        import time
        time.sleep(2)

        # TODO : Effectuer le chargement, la transformation et l’inférence MONAI ici

        findings = {
            "study_id": study_id,
            "modality": modality,
            "model_version": model_version,
            "processed_at": datetime.utcnow().isoformat(),
            "findings": {
                "lesion_count": 3,
                "largest_lesion_mm": 21.5,
                "total_volume_mm3": 4500,
                "locations": ["right_upper_lobe", "left_lower_lobe"],
                "confidence_scores": {
                    "detection": 0.92,
                    "segmentation": 0.88
                },
                "measurements": [
                    {
                        "id": "lesion_1",
                        "location": "right_upper_lobe",
                        "longest_diameter_mm": 21.5,
                        "shortest_diameter_mm": 15.2,
                        "volume_mm3": 2100,
                        "characteristics": {
                            "density_hu": 35.2,
                            "texture": "heterogeneous"
                        }
                    },
                    {
                        "id": "lesion_2",
                        "location": "left_lower_lobe",
                        "longest_diameter_mm": 12.3,
                        "shortest_diameter_mm": 9.8,
                        "volume_mm3": 1200,
                        "characteristics": {
                            "density_hu": 28.7,
                            "texture": "homogeneous"
                        }
                    }
                ]
            },
            "quality_metrics": {
                "image_quality": "good",
                "artifacts": "minimal",
                "coverage": "complete"
            }
        }

        inference_jobs[study_id] = {
            "status": "completed",
            "findings": findings,
            "completed_at": datetime.utcnow()
        }

        # Nettoyage du fichier temporaire
        try:
            os.unlink(file_path)
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Erreur de traitement en arrière‑plan : {str(e)}")
        inference_jobs[study_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow()
        }

@app.get("/infer/status/{study_id}")
async def get_inference_status(study_id: str):
    """Vérifier l'état d'un job d'inférence."""
    job = inference_jobs.get(study_id)
    if not job:
        raise HTTPException(404, f"Aucun job trouvé pour l'étude {study_id}")

    return {
        "study_id": study_id,
        "status": job["status"],
        "timestamp": job.get("completed_at") or job.get("failed_at")
    }

@app.get("/infer/results/{study_id}")
async def get_inference_results(study_id: str):
    """Obtenir les résultats d'inférence."""
    job = inference_jobs.get(study_id)
    if not job or job["status"] != "completed":
        raise HTTPException(404, f"Aucun résultat terminé pour l'étude {study_id}")

    return job["findings"]

# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #

@app.post("/report/generate")
async def generate_report(request: ReportRequest):
    """Générer un rapport radiologique assisté par GPT‑5 en utilisant les résultats et le contexte clinique."""
    try:
        report_id = str(uuid.uuid4())

        # Simuler l’appel à GPT‑5
        report_data = await simulate_gpt5_report(request.findings, request.clinical_context)

        report_response = ReportResponse(
            report_id=report_id,
            status="completed",
            structured_report=report_data.get("structured_report"),
            radiology_narrative=report_data.get("radiology_narrative"),
            patient_summary=report_data.get("patient_summary"),
            generated_at=datetime.utcnow()
        )

        # Stockage en mémoire
        reports_db[report_id] = {
            "request": request.dict(),
            "response": report_response.dict(),
            "created_at": datetime.utcnow()
        }

        return report_response

    except Exception as e:
        logger.error(f"Erreur de génération de rapport : {str(e)}")
        raise HTTPException(500, f"Échec de la génération du rapport : {str(e)}")

async def simulate_gpt5_report(findings: Dict[str, Any], clinical_context: Optional[Dict[str, Any]] = None):
    """Simuler la génération d’un rapport GPT‑5."""
    # Extraction des informations clés
    lesion_count = findings.get("findings", {}).get("lesion_count", 0)
    largest_lesion = findings.get("findings", {}).get("largest_lesion_mm", 0)
    locations = findings.get("findings", {}).get("locations", [])

    # Contexte clinique
    age = clinical_context.get("age") if clinical_context else "inconnu"
    smoking = clinical_context.get("smoking_history") if clinical_context else "inconnu"

    structured_report = {
        "impression": "Nodules pulmonaires multiples détectés",
        "findings": {
            "nodule_count": lesion_count,
            "largest_nodule_mm": largest_lesion,
            "locations": locations,
            "characteristics": "Nodules de densité mixte",
            "recommendations": [
                "CT de suivi dans 3-6 mois",
                "Considérer le PET‑CT si indiquée cliniquement"
            ]
        },
        "confidence_level": "élevée"
    }

    radiology_narrative = f"""
EXAM: CT CHEST SANS CONTRASTE

OBSERVATIONS :
Multiples nodules pulmonaires sont identifiés, le plus grand mesurant {largest_lesion} mm dans le lobe supérieur droit. 
Un total de {lesion_count} nodules sont présents, répartis dans {', '.join(locations)}.

Les nodules affichent des caractéristiques d'atténuation hétérogènes. Aucun ganglion lymphatique associé 
ou effusion pleural n'est identifié. Les voies respiratoires sont perçantes. La structure cardiovasculaire est normale.

IMPRESSION :
Nodules pulmonaires multiples, probablement bénins compte tenu du contexte clinique du patient. Recommander un 
CT de suivi dans 3‑6 mois pour évaluer les changements d'intervalle. Une corrélation clinique est conseillée.
""".strip()

    patient_summary = f"""
Votre récent scanner montre quelques petites taches dans vos poumons appelées nodules. Il s'agit en grande partie d'une observation bénigne. Le plus grand mesure environ {largest_lesion} mm (à peu près la taille d'une petite tomate). 

Votre médecin recommandera probablement un autre examen dans quelques mois pour voir si ces nodules changent. Dans la plupart des cas, ils restent stables ou disparaissent d’eux‑mêmes. Veuillez en discuter lors de votre prochaine consultation.
""".strip()

    return {
        "structured_report": structured_report,
        "radiology_narrative": radiology_narrative,
        "patient_summary": patient_summary
    }

@app.get("/report/{report_id}")
async def get_report(report_id: str):
    """Récupérer un rapport généré."""
    report = reports_db.get(report_id)
    if not report:
        raise HTTPException(404, f"Rapport {report_id} non trouvé")

    return report

@app.get("/reports")
async def list_reports(limit: int = 10, offset: int = 0):
    """Lister les rapports générés."""
    reports = list(reports_db.values())[offset:offset+limit]
    return {
        "total": len(reports_db),
        "reports": reports,
        "limit": limit,
        "offset": offset
    }

# --------------------------------------------------------------------------- #
# Lancement
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )