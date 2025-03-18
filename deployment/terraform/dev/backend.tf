terraform {
  backend "gcs" {
    bucket = "qwiklabs-gcp-03-3716b0e74aea-terraform-state"
    prefix = "dev"
  }
}
