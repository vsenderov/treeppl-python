import json
from tempfile import TemporaryDirectory
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import tempfile


from .exceptions import CompileError, InferenceError
from .serialization import from_json, to_json
import shutil  # Ensure shutil is imported


class Model:
    def __init__(
        self, source=None, filename=None, method="smc-bpf", samples=1_000, **kwargs
    ):
        print("Initializing Model...")  # Debugging output
        self.temp_dir = TemporaryDirectory(prefix="treeppl_")
        print(f"Temporary directory created at {self.temp_dir.name}")  # Debugging output
        if filename:
            source = open(filename).read()
            print(f"Source code read from file: {filename}")  # Debugging output
        if not source:
            raise CompileError("No source code to compile.")
        with open(self.temp_dir.name + "/__main__.tppl", "w") as f:
            f.write(source)
            print("Source code written to temporary file.", self.temp_dir.name)  # Debugging output
        args = [
            "tpplc",
            "__main__.tppl",
            "-m",
            method,
        ]
        for k, v in kwargs.items():
            args.append(f"--{k.replace('_', '-')}")
            if v is not True:
                args.append(str(v))
        print(f"Compiling with arguments: {' '.join(args)}")  # Debugging output
        with Popen(
            args=args, cwd=self.temp_dir.name, stdout=PIPE, stderr=STDOUT
        ) as proc:
            proc.wait()
            if proc.returncode != 0:
                output = proc.stdout.read().decode("utf-8")
                output = output.replace("__main__.tppl", "source code")
                raise CompileError(f"Could not compile the TreePPL model:\n{output}")
            else:
                print("Compilation successful.")  # Debugging output
        self.set_samples(samples)

    def set_samples(self, samples):
        self.samples = samples

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.temp_dir.cleanup()

    def __call__(self, **kwargs):
        with open(self.temp_dir.name + "/input.json", "w") as f:
            to_json(kwargs or {}, f)
        args = [
            f"{self.temp_dir.name}/out",
            f"{self.temp_dir.name}/input.json",
            str(self.samples),
        ]
        # Copy input.json and executable to /tmp directory
        shutil.copy(args[1], "/tmp/input.json")
        shutil.copy(args[0], "/tmp/out")
        with Popen(args=args, stdout=PIPE) as proc:
            return InferenceResult(proc.stdout)


class InferenceResult:
    def __init__(self, stdout):
        self.stdout = stdout  # Assign stdout to self.stdout immediately at the beginning
        try:
            result = from_json(stdout)
        except json.decoder.JSONDecodeError:
            dump_file_path = self.dump_stdout()  # Call method to dump stdout and get file path
            raise InferenceError(f"Could not parse the output from TreePPL. Stdout dumped to {dump_file_path}.")
        self.samples = result.get("samples", [])
        self.weights = np.array(result.get("weights", []))
        self.nweights = np.exp(self.weights)
        self.norm_const = result.get("normConst", np.nan)

    def subsample(self, size=1):
        idx = np.random.choice(len(self.nweights), size, p=self.nweights)
        return [self.samples[i] for i in idx]
    
    def getsample(self):
        idx=range(len(self.nweights))
        return [self.samples[i] for i in idx]

    def dump_stdout(self):
        # Use the tempfile module to create a unique temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode='w', dir='/tmp', prefix='inference_stdout_', suffix='.txt') as tmp_file:
            tmp_file.write(self.stdout)
            return tmp_file.name  # Return the path to the temporary file for reference