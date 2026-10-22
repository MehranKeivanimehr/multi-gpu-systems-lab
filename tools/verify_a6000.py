"""Replay the shipped A6000 calibration and matched-cap throughput evidence."""
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from a6000_evidence import EvidenceError, inference, replay


def main():
    try:
        result = {'calibration':replay(),'inference':inference()}
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result['calibration']['all_pass'] else 1
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as e:
        print('REFUSED: '+str(e),file=sys.stderr)
        return 2


if __name__=='__main__': raise SystemExit(main())
