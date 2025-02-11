import os
best_model_path = "saved_models/best_resnet_artstyle.pth"

if os.path.exists(best_model_path):
    modified_time = os.path.getmtime(best_model_path)
    from datetime import datetime
    print("Best Model Last Modified:", datetime.fromtimestamp(modified_time))
else:
    print("Best model not found.")