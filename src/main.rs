// src/main.rs

use std::{net::SocketAddr, str::FromStr};

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "moestream", version, about = "Storage-aware MoE inference runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the OpenAI-compatible HTTP API scaffold.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8000")]
        address: String,
    },
    /// Print the architecture roadmap and current implementation status.
    Status,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    match Cli::parse().command {
        Command::Serve { address } => {
            let address = SocketAddr::from_str(&address)?;
            moestream::server::serve(address).await?;
        }
        Command::Status => {
            println!(
                "MoEStream {}: storage-aware cache foundation complete; model adapters and tensor execution pending.",
                env!("CARGO_PKG_VERSION")
            );
        }
    }

    Ok(())
}
