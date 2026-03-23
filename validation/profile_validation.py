# function to validate input
def validate_profile(profile):
    from dataclasses import asdict
    
    for key, value in asdict(profile).items():
        # C fields can be 0 (not answered), others must be 1-4
        if key.startswith("C"):
            if value not in [0, 1, 2, 3, 4]:
                return False
        else:
            if value not in [1, 2, 3, 4]:
                return False
    return True