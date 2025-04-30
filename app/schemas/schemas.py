from pydantic import BaseModel
from typing import List, Dict, Any

class AITrainingData(BaseModel):
    """
    Defines the structure of the incoming JSON.
    """
    type: str
    id: str
    name: str
    description: str
    version: str
    amountOfTrainingData: int
    createdTime: str
    providers: List[str]
    bands: List[Dict[str, List[Dict[str, str]]]]
    classes: List[Dict[str, Any]]
    numberOfClasses: int
    tasks: List[Dict[str, Any]]
    data: List[Dict[str, Any]]