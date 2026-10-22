import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import anthropic
from pathlib import Path
import random
@dataclass

class ClinicalNote:
    """Represents a single clinical note with metadata."""
    stage: int
    note_text: str

class ClinicalNotesGenerator:
    """Generates synthetic clinical notes using Anthropic's Claude model."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the generator with Anthropic API key."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided or set as ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Template for generating clinical notes
        self.prompt_template = """You are a primary care physician documenting an initial clinical encounter with an older adult patient presenting with gradually worsening cognitive and language-related concerns. The symptoms are consistent with what might be observed in the early, middle, or late stages of a progressive condition affecting language and behavior. THIS IS SEMANTIC VARIANT PRIMARY PROGRESSIVE APHASIA.

The patient is in stage {stage}.

Your task is to write a **detailed, realistic clinical note** that reflects how this visit might be documented in a primary care setting.

Think babout what information a primary care physician would want to know about the patient and would have access to.

Include the following sections in the note:
1. **Patient demographics** – Generate a plausible age between 65–95, gender, and a realistic patient ID in the format P#### (e.g., P0472)
2. **Chief complaint and history of present illness** – Describe the main concerns prompting the visit and the story behind their development
3. **Patient's description of symptoms** – Use everyday language to capture what the patient (or accompanying family member) says, including examples of how symptoms interfere with daily life (e.g., conversations, errands, reading, relationships)
4. **Clinical observations** – Briefly summarize what the physician notes during the visit (e.g., communication style, affect, orientation, responses)
5. **Relevant medical and family history** – Mention any conditions or life factors that may be contributing to the current situation

**Guidelines for tone and content:**
- Write in a natural, clinical voice—avoid overly dramatic or generic language
- Use plausible quotes or paraphrased speech from the patient or caregiver to add realism
- Symptoms should unfold subtly and progressively over time
- Avoid clinical terminology, diagnoses, or references to brain structures
- Do not include test results, imaging, or specialist findings—this is a first visit
- Focus on observable behavior, functional impact, and context
- Choose a few (not all) common symptom types to emphasize, based on the stage

**Common symptoms to consider (choose selectively and describe organically):**
- Trouble finding words or keeping up with conversations
- Difficulty recognizing familiar names or objects
- Mild forgetfulness or disorientation
- Changes in social behavior or emotional reactions
- Struggles with planning, routine tasks, or using household items
- Frustration, embarrassment, or anxiety related to these changes

Write the clinical note as it would appear in a real primary care chart. Do not over do it. Be realistic to what a clinical note would look like.
"""
    def generate_note(self, stage: int) -> ClinicalNote:
        """Generate a single clinical note using Claude."""
        
        prompt = self.prompt_template.format(stage=stage)
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0,
            system="You are a clinical neurologist writing detailed clinical notes about patients' visits. Focus on patient-reported symptoms and their impact on daily life. Generate realistic patient demographics and ID randomly.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        note_text = response.content[0].text
        
        return ClinicalNote(
            stage=stage,
            note_text=note_text
        )
    
    def generate_notes_batch(self, num_notes: int) -> List[ClinicalNote]:
        """Generate a batch of clinical notes with varying characteristics."""
        
        notes = []
        for _ in range(num_notes):
            # Generate note with random stage
            note = self.generate_note(stage=random.randint(1, 6))
            notes.append(note)
            
        return notes
    
    def save_notes_to_file(self, notes: List[ClinicalNote], output_path: str):
        """Save generated notes to a text file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for note in notes:
                f.write(f"Stage: {note.stage}\n")
                f.write("\nClinical Note:\n")
                f.write(note.note_text)
                f.write("\n" + "="*80 + "\n\n")

def main():
    """Example usage of the ClinicalNotesGenerator."""
    generator = ClinicalNotesGenerator()
    
    # Generate a single note
    note = generator.generate_note(stage=random.randint(1, 6))
    
    # Generate a batch of notes
    notes = generator.generate_notes_batch(num_notes=5)
    
    # Save notes to file
    generator.save_notes_to_file(notes, "generated_clinical_notes.txt")

if __name__ == "__main__":
    main() 