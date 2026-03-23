from statistics import mean
from dataclasses import dataclass
from validation.profile_validation import validate_profile

@dataclass
class Profile:
    A1: int
    A2: int
    B1: int
    B2: int
    B3: int
    B4: int
    C1: int = 0
    C2: int = 0
    C3: int = 0
    C4: int = 0

# function to convert response to quantitative score
def calculate_risk(profile):
    if not validate_profile(profile):
        raise ValueError("Invalid Profile Input")
    
    rt = mean([
        profile.A1, 
        profile.A2
    ])

    if all([profile.C1, profile.C2, profile.C3, profile.C4]):
        blt = mean([profile.C1, profile.C2, profile.C3, profile.C4])
    else:
        blt = mean([profile.B1, profile.B2, profile.B3, profile.B4])

    score = ((rt / 4) * 30) + ((blt / 4) * 70)
    return round(score, 2)