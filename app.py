import argparse,json
from pathlib import Path
from model import TriageModel,ROOT
from api import serve

def main():
    parser=argparse.ArgumentParser(description='Classify support tickets or run the local API')
    parser.add_argument('command',choices=['evaluate','predict','serve'])
    parser.add_argument('--text');parser.add_argument('--port',type=int,default=8001)
    parser.add_argument('--train',type=Path,default=ROOT/'data/train.csv')
    parser.add_argument('--test',type=Path,default=ROOT/'data/test.csv')
    args=parser.parse_args();model=TriageModel(args.train)
    if args.command=='serve':serve(lambda body:model.predict(body.get('text')),args.port)
    elif args.command=='predict':print(json.dumps(model.predict(args.text),indent=2))
    else:
        report=model.evaluate(args.test);out=ROOT/'reports/evaluation.json';out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
