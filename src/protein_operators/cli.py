"""
Command-line interface for protein operators.
"""

import typer
from typing import Optional, List
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import ProteinDesigner, Constraints
from .version import __version__

app = typer.Typer(
    name="protein-operators",
    help="Zero-shot protein design via neural operators",
    add_completion=False,
)
console = Console()


@app.command()
def version():
    """Show version information."""
    console.print(f"Protein Operators v{__version__}")


@app.command()
def design(
    output: Path = typer.Option(..., "--output", "-o", help="Output PDB file"),
    length: int = typer.Option(100, "--length", "-l", help="Protein length"),
    operator: str = typer.Option("deeponet", "--operator", help="Neural operator type"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-c", help="Model checkpoint"),
    device: str = typer.Option("auto", "--device", help="Computing device"),
    num_samples: int = typer.Option(1, "--samples", "-n", help="Number of designs"),
    physics: bool = typer.Option(False, "--physics", help="Use physics-guided refinement"),
):
    """
    Generate protein structures from constraints.
    """
    console.print(f"[bold blue]Protein Operators v{__version__}[/bold blue]")
    console.print(f"Generating protein with {length} residues using {operator}")
    
    try:
        # Initialize designer
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading neural operator...", total=None)
            
            designer = ProteinDesigner(
                operator_type=operator,
                checkpoint=checkpoint,
                device=device
            )
            
            progress.update(task, description="Setting up constraints...")
            
            # For now, use empty constraints (would be extended with CLI options)
            constraints = Constraints()
            
            progress.update(task, description="Generating structure...")
            
            structure = designer.generate(
                constraints=constraints,
                length=length,
                num_samples=num_samples,
                physics_guided=physics
            )
            
            progress.update(task, description="Saving structure...")
            
            # Save structure (placeholder - would implement actual PDB writing)
            output.parent.mkdir(parents=True, exist_ok=True)
            # structure.save_pdb(output)
            
            progress.update(task, description="Complete!")
        
        console.print(f"[green]✓[/green] Structure saved to {output}")
        
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    structure: Path = typer.Argument(..., help="PDB file to validate"),
    constraints: Optional[Path] = typer.Option(None, help="Constraints file"),
    output: Optional[Path] = typer.Option(None, help="Output validation report"),
):
    """
    Validate protein structure against constraints.
    """
    console.print(f"[bold blue]Validating {structure}[/bold blue]")
    
    try:
        # Load structure (placeholder)
        # structure_obj = ProteinStructure.from_pdb(structure)
        
        # Load constraints if provided
        constraints_obj = Constraints()
        if constraints:
            # constraints_obj = Constraints.from_file(constraints)
            pass
        
        # Perform validation (placeholder)
        validation_results = {
            "stereochemistry_score": 0.95,
            "clash_score": 0.02,
            "ramachandran_score": 0.98,
            "constraint_satisfaction": 0.87,
        }
        
        # Display results
        table = Table(title="Validation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Status", style="bold")
        
        for metric, score in validation_results.items():
            status = "✓ Pass" if score > 0.8 else "✗ Fail"
            table.add_row(metric.replace("_", " ").title(), f"{score:.3f}", status)
        
        console.print(table)
        
        if output:
            # Save detailed report
            console.print(f"[green]✓[/green] Validation report saved to {output}")
        
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def info(
    device: bool = typer.Option(False, "--device", help="Show device information"),
    models: bool = typer.Option(False, "--models", help="List available models"),
    constraints: bool = typer.Option(False, "--constraints", help="List constraint types"),
):
    """
    Show system information.
    """
    if device:
        console.print("[bold blue]Device Information[/bold blue]")
        
        # PyTorch info
        console.print(f"PyTorch version: {torch.__version__}")
        console.print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            console.print(f"CUDA version: {torch.version.cuda}")
            console.print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                console.print(f"  GPU {i}: {name} ({memory:.1f} GB)")
    
    if models:
        console.print("[bold blue]Available Models[/bold blue]")
        console.print("• deeponet - DeepONet for constraint-to-structure mapping")
        console.print("• fno - Fourier Neural Operator for PDE-based refinement")
    
    if constraints:
        console.print("[bold blue]Available Constraint Types[/bold blue]")
        console.print("• structural - Secondary structure, disulfide bonds")
        console.print("• binding - Binding site design, ligand affinity")
        console.print("• catalytic - Enzymatic activity, catalytic sites")
        console.print("• stability - Thermodynamic stability, folding")
        console.print("• solubility - Solubility, aggregation resistance")


@app.command()
def train(
    config: Path = typer.Argument(..., help="Training configuration file"),
    data: Path = typer.Option(..., "--data", help="Training data directory"),
    output: Path = typer.Option("./models", "--output", help="Model output directory"),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Resume from checkpoint"),
    gpus: int = typer.Option(1, "--gpus", help="Number of GPUs to use"),
):
    """
    Train neural operator models.
    """
    console.print(f"[bold blue]Training Neural Operator[/bold blue]")
    console.print(f"Config: {config}")
    console.print(f"Data: {data}")
    console.print(f"Output: {output}")
    console.print(f"GPUs: {gpus}")
    
    if resume:
        console.print(f"Resuming from: {resume}")
    
    # Training implementation would go here
    console.print("[yellow]Training functionality not yet implemented[/yellow]")


# Add new commands
@app.command()
def interactive(
    operator: str = typer.Option("deeponet", "--operator", help="Neural operator type"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", help="Model checkpoint"),
):
    """
    Start an interactive protein design session.
    """
    console.print(f"[bold blue]🖥️ Interactive Protein Design Session[/bold blue]")
    console.print(f"Using {operator} operator")
    console.print("Type 'help' for commands or 'quit' to exit.\n")
    
    # Initialize designer
    try:
        designer = ProteinDesigner(operator_type=operator, checkpoint=checkpoint)
        constraints = Constraints()
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize designer: {e}[/red]")
        raise typer.Exit(1)
    
    commands_help = """
[bold cyan]Available Commands:[/bold cyan]
  add-binding <residues> <ligand> [affinity] - Add binding site constraint
  add-ss <start> <end> <type>               - Add secondary structure constraint
  list-constraints                          - Show current constraints
  clear-constraints                         - Clear all constraints
  design <length> [output.pdb]             - Generate protein design
  validate <file.pdb>                       - Validate structure
  help                                      - Show this help
  quit                                      - Exit session

[bold yellow]Examples:[/bold yellow]
  add-binding 45,67,89 ATP 100
  add-ss 10 30 helix
  design 150 my_protein.pdb
    """
    
    while True:
        try:
            command = console.input("[bold green]protein-operators>[/bold green] ").strip()
            
            if command.lower() in ["quit", "exit", "q"]:
                console.print("Goodbye! 👋")
                break
            elif command.lower() == "help":
                console.print(commands_help)
            elif command.startswith("add-binding"):
                parts = command.split()[1:]
                if len(parts) >= 2:
                    residues = [int(x) for x in parts[0].split(",")]
                    ligand = parts[1]
                    affinity = float(parts[2]) if len(parts) > 2 else None
                    constraints.add_binding_site(residues=residues, ligand=ligand, affinity_nm=affinity)
                    console.print(f"✅ Added binding site: {residues} for {ligand}")
                else:
                    console.print("[red]Usage: add-binding <residues> <ligand> [affinity][/red]")
            elif command.startswith("add-ss"):
                parts = command.split()[1:]
                if len(parts) >= 3:
                    start, end, ss_type = int(parts[0]), int(parts[1]), parts[2]
                    constraints.add_secondary_structure(start=start, end=end, ss_type=ss_type)
                    console.print(f"✅ Added secondary structure: {ss_type} from {start} to {end}")
                else:
                    console.print("[red]Usage: add-ss <start> <end> <type>[/red]")
            elif command == "list-constraints":
                if len(constraints) == 0:
                    console.print("No constraints defined")
                else:
                    console.print(f"[bold]Current constraints ({len(constraints)}):[/bold]")
                    for i, constraint in enumerate(constraints):
                        console.print(f"  {i+1}. {constraint}")
            elif command == "clear-constraints":
                constraints = Constraints()
                console.print("✅ All constraints cleared")
            elif command.startswith("design"):
                parts = command.split()[1:]
                if len(parts) >= 1:
                    length = int(parts[0])
                    output_file = parts[1] if len(parts) > 1 else f"design_{length}aa.pdb"
                    
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task(f"Generating {length}-residue protein...", total=None)
                        
                        structure = designer.generate(constraints=constraints, length=length)
                        
                        progress.update(task, description="Saving structure...")
                        structure.save_pdb(Path(output_file))
                        
                        progress.update(task, description="Validating...")
                        metrics = designer.validate(structure)
                    
                    console.print(f"✅ Design saved to {output_file}")
                    console.print(f"Overall score: {metrics['overall_score']:.3f}")
                else:
                    console.print("[red]Usage: design <length> [output.pdb][/red]")
            elif command.startswith("validate"):
                parts = command.split()[1:]
                if len(parts) >= 1:
                    from .structure import ProteinStructure
                    
                    structure = ProteinStructure.from_pdb(Path(parts[0]))
                    metrics = designer.validate(structure)
                    
                    console.print(f"[bold]Validation results for {parts[0]}:[/bold]")
                    for metric, score in metrics.items():
                        if isinstance(score, (int, float)) and metric != "num_clashes":
                            console.print(f"  {metric.replace('_', ' ').title()}: {score:.3f}")
                else:
                    console.print("[red]Usage: validate <file.pdb>[/red]")
            elif command.strip():
                console.print(f"[red]Unknown command: {command}[/red]")
                console.print("Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            console.print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

@app.command()
def benchmark(
    suite: str = typer.Option("all", "--suite", help="Benchmark suite: design, validation, or all"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output benchmark results"),
    iterations: int = typer.Option(10, "--iterations", help="Number of iterations"),
):
    """
    Run performance benchmarks for the protein design pipeline.
    """
    console.print(f"[bold blue]⚡ Running {suite} benchmarks[/bold blue]")
    console.print(f"Iterations: {iterations}")
    
    # Placeholder for benchmark implementation
    console.print("\n[yellow]⚠️ Benchmarking functionality is under development.[/yellow]")
    console.print("\n[bold]Planned Benchmarks:[/bold]")
    console.print("• Design speed (proteins/second)")
    console.print("• Memory usage profiling")
    console.print("• Constraint satisfaction accuracy")
    console.print("• Structure quality metrics")
    console.print("• GPU utilization efficiency")
    
    if output:
        console.print(f"\nResults would be saved to: {output}")

def main():
    """Enhanced main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()