"""Console script for alethia."""

import typer
from rich.console import Console

import alethia

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for alethia."""
    console.print("Replace this message by putting your code into " "alethia.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")


if __name__ == "__main__":
    app()
