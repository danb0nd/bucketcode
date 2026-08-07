//! bucketcode — agentic coding over bucketlang.

use bucketcode::budget::{Budget, Guard};
use bucketcode::index::search;
use bucketcode::llm::{Client, Config};
use bucketcode::strategy::{self, Strategy};
use bucketlang::compile::{compile_file, BuildProfile, CompileOptions};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(name = "bucketcode", version, about = "Edit one bucket at a time, not whole files")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Find buckets by describing what you want, in English.
    Find {
        file: PathBuf,
        /// What you're looking for, in your own words.
        query: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
    },

    /// Show exactly what would be sent to the model, and what it costs.
    ///
    /// No API call. Use it to compare strategies before spending anything.
    Context {
        file: PathBuf,
        #[arg(long)]
        bucket: String,
        #[arg(long, default_value = "auto")]
        strategy: String,
        #[arg(long, default_value_t = 1)]
        depth: usize,
    },

    /// Measure context cost across program sizes. No API calls.
    Bench {
        /// Bucket counts to sweep.
        #[arg(long, value_delimiter = ',', default_value = "0,2,5,10,20,40,80,160,320")]
        sizes: Vec<usize>,
        /// Write CSV here.
        #[arg(long)]
        csv: Option<PathBuf>,
    },

    /// Open the local web UI.
    ///
    /// Everything except the agent works without an API key.
    Serve {
        file: PathBuf,
        #[arg(long, default_value_t = 7878)]
        port: u16,
    },

    /// Edit a bucket with a real model.
    Edit {
        file: PathBuf,
        /// What to change, in English. Also used to find the bucket if
        /// `--bucket` is not given.
        goal: String,
        /// Skip retrieval and edit this bucket directly.
        #[arg(long)]
        bucket: Option<String>,
        #[arg(long, default_value = "auto")]
        strategy: String,
        /// How many tool-use turns the agent may take.
        ///
        /// Not a retry count: one turn is one model call, and a task that needs
        /// to read, edit, fix a test, and re-edit needs several. The default was
        /// 3, which silently truncated multi-step work before the spend limits
        /// ever came into play.
        #[arg(long, default_value_t = 12)]
        max_turns: u32,
        #[arg(long, default_value_t = 1)]
        depth: usize,
        /// Persist the edit if every gate passes.
        #[arg(long)]
        write: bool,

        // ---- spend limits ----
        /// Hard ceiling on API calls for this run.
        #[arg(long, default_value_t = 8)]
        max_requests: u32,
        /// Hard ceiling on estimated spend, in USD.
        #[arg(long, default_value_t = 1.0)]
        max_usd: f64,
        /// Hard ceiling on output tokens across the run.
        #[arg(long, default_value_t = 32_000)]
        max_output_tokens: u32,

        #[arg(long, default_value = bucketcode::llm::DEFAULT_MODEL)]
        model: String,
        /// low | medium | high | xhigh | max
        #[arg(long, default_value = "high")]
        effort: String,
        /// Print the request and stop, without calling the API.
        #[arg(long)]
        dry_run: bool,
        /// Command execution: auto (allowlisted commands run), ask, or never.
        #[arg(long, default_value = "auto")]
        commands: String,
        /// Also run the conventional whole-file arm and print a token comparison.
        #[arg(long)]
        compare: bool,
    },
}

fn opts() -> CompileOptions {
    CompileOptions {
        strict: true,
        profile: BuildProfile::Dev,
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Find { file, query, limit } => {
            let compiled = compile_file(&file, opts()).map_err(|e| e.to_string())?;
            let matches = search(&compiled.registry, &query, limit);
            if matches.is_empty() {
                println!("no buckets matched \"{query}\"");
                println!("\ntry different words, or list what exists with:  bkt inspect {} --labels", file.display());
                return Ok(());
            }
            for m in &matches {
                println!(
                    "{:>6.2}  {}  {}{}\n        \"{}\"\n        matched: {}",
                    m.score,
                    m.address,
                    m.label.clone().unwrap_or_default(),
                    m.contract,
                    m.desc,
                    m.matched.join(", ")
                );
            }
            Ok(())
        }

        Commands::Bench { sizes, csv } => {
            let rows = bucketcode::bench::sweep(&sizes)?;
            println!(
                "{:>8} {:>12} {:>12} {:>10} {:>9} {:>10}",
                "buckets", "file_tok", "subgraph_tok", "overhead", "ctx x", "effective x"
            );
            for r in &rows {
                println!(
                    "{:>8} {:>12} {:>12} {:>10} {:>8.2}x {:>9.2}x",
                    r.buckets, r.file_tokens, r.subgraph_tokens,
                    r.agent_overhead, r.context_ratio, r.effective_ratio
                );
            }
            if let Some(path) = csv {
                std::fs::write(&path, bucketcode::bench::to_csv(&rows))
                    .map_err(|e| e.to_string())?;
                eprintln!("\nwrote {}", path.display());
            }
            Ok(())
        }

        Commands::Serve { file, port } => bucketcode::serve::serve(file, port),

        Commands::Context {
            file,
            bucket,
            strategy: strat,
            depth,
        } => {
            let compiled = compile_file(&file, opts()).map_err(|e| e.to_string())?;
            let src = std::fs::read_to_string(&file).map_err(|e| e.to_string())?;
            let s: Strategy = strat.parse()?;
            let rendered = strategy::build(s, &compiled.registry, &src, &bucket, depth)?;

            println!("{}", rendered.text);
            eprintln!("---");
            eprintln!(
                "strategy: {}  |  {} tok  |  alternative: {} tok  |  {:.2}x",
                rendered.strategy.as_str(),
                rendered.tokens,
                rendered.alternative_tokens,
                rendered.savings_ratio()
            );
            Ok(())
        }

        Commands::Edit {
            file,
            goal,
            bucket,
            strategy: strat,
            max_turns,
            depth,
            write,
            max_requests,
            max_usd,
            max_output_tokens,
            model,
            effort,
            dry_run,
            commands,
            compare,
        } => {
            let approval = match commands.as_str() {
                "auto" => bucketcode::sandbox::ApprovalMode::Auto,
                "ask" => bucketcode::sandbox::ApprovalMode::Ask,
                "never" => bucketcode::sandbox::ApprovalMode::Never,
                other => return Err(format!("unknown --commands mode '{other}'")),
            };
            let compiled = compile_file(&file, opts()).map_err(|e| e.to_string())?;
            let src = std::fs::read_to_string(&file).map_err(|e| e.to_string())?;

            // A named bucket becomes a hint; otherwise the agent finds its own
            // target with find_buckets, which is the flow being tested.
            let goal = match &bucket {
                Some(b) => format!("{goal}\n\n(The relevant bucket is `{b}`.)"),
                None => goal.clone(),
            };
            let _ = &compiled;

            if dry_run {
                let target = bucket.clone().unwrap_or_else(|| "main".into());
                let s: Strategy = strat.parse()?;
                let rendered = strategy::build(s, &compiled.registry, &src, &target, depth)?;
                println!("{}", bucketcode::agent::SYSTEM);
                println!("\n================ tools ================\n");
                for t in bucketcode::agent::tools() {
                    println!("- {}: {}", t.name, t.description);
                }
                println!("\n================ task ================\n{goal}");
                eprintln!("---");
                eprintln!(
                    "reference: subgraph {} tok vs whole file {} tok ({:.2}x) | no API call made",
                    rendered.tokens,
                    rendered.alternative_tokens,
                    rendered.savings_ratio()
                );
                return Ok(());
            }

            let outcome_model = model.clone();
            let effort_for_baseline = effort.clone();
            let cfg = Config::from_env()?
                .with_model(model)
                .with_effort(Some(effort))
                .normalized();
            let budget = Budget {
                max_requests,
                max_input_tokens: 200_000,
                max_output_tokens,
                max_usd,
            };
            eprintln!(
                "limits: {} request(s), {} output tok, ${:.2} — the run aborts at any of these\n",
                budget.max_requests, budget.max_output_tokens, budget.max_usd
            );

            let guard = Guard::new(budget, cfg.model.clone().as_str());
            let client = Client::new(cfg);

            let run = bucketcode::agent::run(
                &client,
                guard,
                &file,
                &src,
                &goal,
                max_turns as usize,
                approval,
            )?;

            if run.ok && write {
                if let Some(new_src) = &run.source {
                    std::fs::write(&file, new_src).map_err(|e| e.to_string())?;
                    eprintln!("wrote {}", file.display());
                }
            }
            eprintln!(
                "\n{} turn(s), {} tool call(s) | {} | whole file would have been {} tok",
                run.turns,
                run.tool_calls.len(),
                run.spend.summary(),
                run.whole_file_tokens
            );
            let outcome = run;
            if compare {
                eprintln!("\n--- running the conventional whole-file arm for comparison ---");
                let base_cfg = Config::from_env()?
                    .with_model(outcome_model.clone())
                    .with_effort(Some(effort_for_baseline.clone()))
                    .normalized();
                let base_guard = Guard::new(
                    Budget { max_requests: 2, max_input_tokens: 200_000,
                             max_output_tokens: 16_000, max_usd },
                    &base_cfg.model,
                );
                let base_client = Client::new(base_cfg);
                match bucketcode::agent::run_baseline(
                    &base_client, base_guard, &file, &src, &goal,
                ) {
                    Ok(b) => {
                        let a = &outcome.spend;
                        let bs = &b.spend;
                        eprintln!();
                        eprintln!("{:<22} {:>14} {:>14}", "", "buckets", "whole file");
                        eprintln!("{:<22} {:>14} {:>14}", "input tokens", a.input_tokens, bs.input_tokens);
                        eprintln!("{:<22} {:>14} {:>14}", "output tokens", a.output_tokens, bs.output_tokens);
                        eprintln!("{:<22} {:>14} {:>14}", "cached input", a.cache_read_tokens, bs.cache_read_tokens);
                        eprintln!("{:<22} {:>13.4} {:>13.4}", "usd", a.usd, bs.usd);
                        eprintln!("{:<22} {:>14} {:>14}", "api calls", a.requests, bs.requests);
                        eprintln!("{:<22} {:>14} {:>14}", "compiles",
                                  if outcome.ok {"yes"} else {"-"}, if b.compiles {"yes"} else {"NO"});
                        eprintln!("{:<22} {:>14} {:>14}", "tests pass",
                                  if outcome.ok {"yes"} else {"-"}, if b.tests_pass {"yes"} else {"NO"});
                        eprintln!("{:<22} {:>14} {:>14}", "atomicity provable", "yes", "no");
                        eprintln!("{:<22} {:>14} {:>14}", "lines rewritten", "1 bucket",
                                  format!("{}", b.changed_lines));
                        let ratio = bs.usd / a.usd.max(1e-9);
                        eprintln!();
                        eprintln!("whole file cost {:.2}x the bucket path on this run", ratio);
                        if let Some(e) = &b.error {
                            eprintln!("baseline error: {}", e.lines().next().unwrap_or(""));
                        }
                    }
                    Err(e) => eprintln!("baseline arm failed: {e}"),
                }
            }
            println!("{}", serde_json::to_string_pretty(&outcome).unwrap());
            if outcome.ok {
                Ok(())
            } else {
                Err(format!("no edit was made in {} turn(s)", outcome.turns))
            }
        }
    }
}
