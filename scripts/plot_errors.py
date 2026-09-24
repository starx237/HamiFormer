from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hamiformer.visualization.error_curves import main
if __name__ == '__main__':
    main()
