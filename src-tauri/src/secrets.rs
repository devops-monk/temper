//! The Hugging Face token, kept in the OS credential store (macOS Keychain, Windows Credential
//! Manager, the Secret Service on Linux) rather than in a file. The engine gets it as HF_TOKEN.

use keyring::Entry;

const SERVICE: &str = "app.temper.studio";
const ACCOUNT: &str = "huggingface-token";

fn entry() -> Result<Entry, String> {
    Entry::new(SERVICE, ACCOUNT).map_err(|e| e.to_string())
}

pub fn hf_token() -> Option<String> {
    entry().ok()?.get_password().ok().filter(|t| !t.is_empty())
}

pub fn set_hf_token(token: &str) -> Result<(), String> {
    entry()?.set_password(token).map_err(|e| format!("Temper couldn't save the token to your system's keychain ({e}). It works until you quit Temper."))
}

pub fn delete_hf_token() -> Result<(), String> {
    match entry()?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}
